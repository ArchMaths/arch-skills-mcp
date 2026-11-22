"""Continuous depth queries on already-generated hair surfaces (no generator I/O).

``CrownContact(surfs).depth(x, z)`` returns the outermost actual surface depth,
not a dilated raster. Missing X/Z projections return NaN. Inputs broadcast and
scalar queries return floats. Surface tuples use sheetfit's (name, colour, f,
limits) convention; f must accept numpy arrays.
"""
import numpy as np


def _hc_points(f, u, v):
    return np.stack(np.broadcast_arrays(*f(u, v)), axis=-1)


class CrownContact:
    """Vectorised, multi-start inversion of the X/Z projection.

    A spatial seed table avoids comparing each query with an entire sample
    cloud. Each cell retains both transverse extrema, so folded projections
    can have two initial guesses. Newton is evaluated only inside each patch's
    sampled X/Z bounds. All valid patch solutions participate in the upper
    envelope; there is no arbitrary fixed-y fallback.

    ``residual`` is an X/Z distance, not a depth tolerance. At nearly edge-on
    projections depth is ill-conditioned even with a small residual; callers
    should use native surface derivatives for root jets. No finite-difference
    derivatives of a raster are used here.
    """

    def __init__(self, surfs, seed_step=0.025, residual=0.0015, iterations=14,
                 cache_step=0.004, zmin=None):
        if seed_step <= 0 or residual <= 0 or iterations < 1:
            raise ValueError('Positive seed_step/residual/iterations required')
        self.seed_step = float(seed_step)
        self.residual = float(residual)
        self.iterations = int(iterations)
        self.patches = []
        self.cache_step = float(cache_step) if cache_step else 0.0
        self._cache = None
        self.zmin = None if zmin is None else float(zmin)
        self.last_stats = {}
        for name, _, f, limits in surfs:
            lim = np.asarray(limits, float).copy()
            # Hair tubes use the positive-sine crescent as their outer half.
            # hairUnder's transverse parameter is [-1.2,1.2], not a tube angle.
            # Keep full non-tube surfaces (including a cap) if supplied.
            if name.startswith(('back', 'side', 'tuck')):
                if abs((lim[3] - lim[2]) - 2*np.pi) < 0.02:
                    lim[2], lim[3] = 0.0, np.pi
                elif abs((lim[1] - lim[0]) - 2*np.pi) < 0.02:
                    lim[0], lim[1] = 0.0, np.pi
            if not (lim[1] > lim[0] and lim[3] > lim[2]):
                continue
            u, v = np.meshgrid(np.linspace(lim[0], lim[1], 17),
                               np.linspace(lim[2], lim[3], 17), indexing='ij')
            with np.errstate(all='ignore'):
                p = _hc_points(f, u, v)
            if not np.isfinite(p).all(-1).any():
                continue
            xz = p[..., (0, 2)]
            lengths = []
            for axis in (0, 1):
                ds = np.linalg.norm(np.diff(xz, axis=axis), axis=-1)
                lengths.append(float(np.nanmax(np.nansum(ds, axis=axis))))
            nu, nv = [int(np.clip(np.ceil(a/(seed_step*0.4))+1, 25, cap))
                      for a, cap in zip(lengths, (450, 180))]
            u, v = np.meshgrid(np.linspace(lim[0], lim[1], nu),
                               np.linspace(lim[2], lim[3], nv), indexing='ij')
            with np.errstate(all='ignore'):
                p = _hc_points(f, u, v).reshape(-1, 3)
            uv = np.c_[u.ravel(), v.ravel()]
            ok = np.isfinite(p).all(-1) & (p[:, 1] > 0)
            p, uv = p[ok], uv[ok]
            if not len(p):
                continue
            xz = p[:, (0, 2)]
            lower, upper = xz.min(0), xz.max(0)
            origin = lower - seed_step
            shape = np.ceil((upper-origin)/seed_step).astype(int) + 2
            ij = np.floor((xz-origin)/seed_step).astype(int)
            cell = ij[:, 0]*shape[1] + ij[:, 1]
            # The first/last transverse samples in a bin seed separate branches.
            order = np.lexsort((uv[:, 1], cell))
            sorted_cell = cell[order]
            first = np.r_[True, sorted_cell[1:] != sorted_cell[:-1]]
            last = np.r_[sorted_cell[1:] != sorted_cell[:-1], True]
            table = np.full((int(np.prod(shape)), 3), -1, dtype=np.int32)
            table[sorted_cell[first], 0] = order[first]
            table[sorted_cell[last], 1] = order[last]
            outer_order = np.lexsort((p[:, 1], cell))
            outer_cell = cell[outer_order]
            outer_last = np.r_[outer_cell[1:] != outer_cell[:-1], True]
            table[outer_cell[outer_last], 2] = outer_order[outer_last]
            self.patches.append(dict(name=name, f=f, lim=lim, uv=uv, xz=xz, y=p[:, 1],
                                     lower=lower, upper=upper, origin=origin,
                                     shape=shape, table=table))

    def _seed(self, patch, xz):
        ij = np.floor((xz-patch['origin'])/self.seed_step).astype(int)
        candidates = []
        nx, nz = patch['shape']
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                i, j = ij[:, 0]+di, ij[:, 1]+dj
                ok = (i >= 0) & (i < nx) & (j >= 0) & (j < nz)
                ids = patch['table'][np.clip(i, 0, nx-1)*nz + np.clip(j, 0, nz-1)]
                candidates.append(np.where(ok[:, None], ids, -1))
        ids = np.concatenate(candidates, axis=1)
        positions = patch['xz'][np.maximum(ids, 0)]
        score = np.sum((positions-xz[:, None, :])**2, axis=-1)
        score[ids < 0] = np.inf
        row = np.arange(len(xz))
        first = np.argmin(score, axis=1)
        first_ids = ids[row, first]
        uv = patch['uv'][np.maximum(first_ids, 0)]
        # Second start must be genuinely different, not the same sample repeated
        # by neighbouring bins. This is important near an X/Z projection fold.
        span = patch['lim'][[1, 3]]-patch['lim'][[0, 2]]
        distance = np.linalg.norm((patch['uv'][np.maximum(ids, 0)]-uv[:, None, :])/span,
                                  axis=-1)
        second_score = np.where(distance > 0.035, score, np.inf)
        second = np.argmin(second_score, axis=1)
        second_ids = ids[row, second]
        valid_first = np.isfinite(score[row, first])
        valid_second = np.isfinite(second_score[row, second])
        # Do not send starts from distant cells to a patch that does not cover
        # the query; a second basin within the local seed neighbourhood is enough.
        outer_score = np.where((ids >= 0) & (score < (2*self.seed_step)**2),
                               patch['y'][np.maximum(ids, 0)], -np.inf)
        outer = np.argmax(outer_score, axis=1)
        outer_ids = ids[row, outer]
        outer_uv = patch['uv'][np.maximum(outer_ids, 0)]
        # Periodic cap extrema (-pi,+pi) can be the very same rear branch;
        # explicitly preserve a maximum-y seed, not only parameter extrema.
        valid_outer = np.isfinite(outer_score[row, outer])
        return [(uv, valid_first),
                (patch['uv'][np.maximum(second_ids, 0)], valid_second),
                (outer_uv, valid_outer)]

    def _invert(self, patch, xz, initial, with_gradient=False):
        f, lim = patch['f'], patch['lim']
        uv = initial.copy()
        lo, hi = lim[[0, 2]], lim[[1, 3]]
        span = hi-lo
        eps = np.maximum(span*1e-5, 1e-7)
        best_error = np.full(len(xz), np.inf)
        best_y = np.full(len(xz), np.nan)
        best_uv = uv.copy()
        # Retain the best iterate: clipping near a fold can otherwise oscillate
        # between a good iterate and a poor boundary iterate.
        for _ in range(self.iterations):
            with np.errstate(all='ignore'):
                u, v = uv[:, 0], uv[:, 1]
                p = _hc_points(f, u, v)
                err = np.sum((p[:, (0, 2)]-xz)**2, axis=1)
                better = np.isfinite(p).all(1) & (p[:, 1] > 0) & (err < best_error)
                best_error[better], best_y[better] = err[better], p[better, 1]
                best_uv[better] = uv[better]
                if np.all(best_error < 1e-16):
                    break
                du = (_hc_points(f, u+eps[0], v)-_hc_points(f, u-eps[0], v))/(2*eps[0])
                dv = (_hc_points(f, u, v+eps[1])-_hc_points(f, u, v-eps[1]))/(2*eps[1])
                r = p[:, (0, 2)]-xz
                det = du[:, 0]*dv[:, 2]-dv[:, 0]*du[:, 2]
                good = np.isfinite(det) & (np.abs(det) > 1e-12)
                denominator = np.where(good, det, np.inf)
                delta = np.c_[(dv[:, 2]*r[:, 0]-dv[:, 0]*r[:, 1])/denominator,
                              (-du[:, 2]*r[:, 0]+du[:, 0]*r[:, 1])/denominator]
                delta = np.nan_to_num(delta)
                delta = np.clip(delta, -0.25*span, 0.25*span)
                uv = np.clip(uv-delta, lo, hi)
        # Include the last Newton update, not just its predecessor.
        with np.errstate(all='ignore'):
            p = _hc_points(f, uv[:, 0], uv[:, 1])
        err = np.sum((p[:, (0, 2)]-xz)**2, axis=1)
        better = np.isfinite(p).all(1) & (p[:, 1] > 0) & (err < best_error)
        best_error[better], best_y[better] = err[better], p[better, 1]
        best_uv[better] = uv[better]
        y = np.where(best_error < self.residual**2, best_y, np.nan)
        if not with_gradient:
            return y
        u, v = best_uv[:, 0], best_uv[:, 1]
        with np.errstate(all='ignore'):
            du = (_hc_points(f, u+eps[0], v)-_hc_points(f, u-eps[0], v))/(2*eps[0])
            dv = (_hc_points(f, u, v+eps[1])-_hc_points(f, u, v-eps[1]))/(2*eps[1])
            det = du[:, 0]*dv[:, 2]-dv[:, 0]*du[:, 2]
            gx = (du[:, 1]*dv[:, 2]-dv[:, 1]*du[:, 2])/det
            gz = (du[:, 0]*dv[:, 1]-dv[:, 0]*du[:, 1])/det
        valid = np.isfinite(y) & (np.abs(det) > 1e-12)
        return y, np.where(valid, gx, np.nan), np.where(valid, gz, np.nan)

    def depth_exact(self, x, z):
        """Maximum y of converged actual-surface projections, or NaN.

        Arrays broadcast. Unlike a nearest-neighbour/dilated depth grid, returned
        finite values are evaluated on real surface points with X/Z residual
        below ``self.residual``. Two independent starts are tried for folds.
        """
        x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        shape = x.shape
        query = np.c_[x.ravel(), z.ravel()]
        if self.cache_step:
            self._build_cache()
            out = self._native_contact(query)[:, 0]
        else:
            out = np.full(len(query), np.nan)
        finite = np.isfinite(query).all(1)
        attempted = accepted = 0
        for patch in self.patches:
            margin = self.residual
            keep = finite & (query >= patch['lower']-margin).all(1) & (query <= patch['upper']+margin).all(1)
            indices = np.flatnonzero(keep)
            if not len(indices):
                continue
            q = query[indices]
            for seed, ok in self._seed(patch, q):
                if not ok.any():
                    continue
                target = indices[ok]
                y = self._invert(patch, q[ok], seed[ok])
                attempted += len(target)
                accepted += int(np.isfinite(y).sum())
                out[target] = np.fmax(out[target], y)
        self.last_stats = dict(queries=len(query), attempted_starts=attempted,
                               converged_starts=accepted, finite=int(np.isfinite(out).sum()))
        result = out.reshape(shape)
        return float(result) if not shape else result

    def _build_cache(self):
        """Fine piecewise-linear upper envelope, with native UV seeds retained.

        This is a numerical X/Z depth field, not an image or Arch render. Every
        cache node is covered by an actual sampled parameter triangle; empty
        cells are not grown, wrapped, nearest-filled, or assigned a depth floor.
        """
        if self._cache is not None:
            return
        step = self.cache_step
        clouds = [p['xz'] if self.zmin is None else p['xz'][p['xz'][:, 1] >= self.zmin]
                  for p in self.patches]
        clouds = [p for p in clouds if len(p)]
        if not clouds:
            self._cache = dict(origin=np.zeros(2), shape=np.array([2, 2]),
                               height=np.full(4, -np.inf), owner=np.full(4, -1, dtype=np.int16),
                               uv=np.full((4, 2), np.nan), step=step)
            return
        cloud = np.concatenate(clouds)
        lower, upper = cloud.min(0), cloud.max(0)
        if self.zmin is not None:
            lower[1] = self.zmin
        origin = np.floor(lower/step)*step-2*step
        shape = np.ceil((upper-origin)/step).astype(int)+3
        size = int(np.prod(shape))
        height = np.full(size, -np.inf)
        owner = np.full(size, -1, dtype=np.int16)
        native = np.full((size, 2), np.nan)
        for pi, patch in enumerate(self.patches):
            if patch['upper'][1] < lower[1] or patch['lower'][1] > upper[1]:
                continue
            f, lim = patch['f'], patch['lim'].copy()
            a, b = np.meshgrid(np.linspace(lim[0], lim[1], 45),
                               np.linspace(lim[2], lim[3], 41), indexing='ij')
            with np.errstate(all='ignore'):
                p = _hc_points(f, a, b)
            ok = np.isfinite(p).all(-1) & (p[..., 1] > 0) & (p[..., 2] >= lower[1]-step)
            rows = np.flatnonzero(ok.any(1))
            if not len(rows):
                continue
            r0, r1 = max(0, rows[0]-1), min(44, rows[-1]+1)
            lim[0], lim[1] = a[r0, 0], a[r1, 0]
            a, b = np.meshgrid(np.linspace(lim[0], lim[1], 33),
                               np.linspace(lim[2], lim[3], 33), indexing='ij')
            with np.errstate(all='ignore'):
                p = _hc_points(f, a, b)
            lengths = [float(np.nanmax(np.nansum(np.linalg.norm(np.diff(p[..., (0, 2)], axis=axis),
                                                                             axis=-1), axis=axis)))
                       for axis in (0, 1)]
            nu, nv = [int(np.clip(np.ceil(v/(step*0.55))+1, 33, cap))
                      for v, cap in zip(lengths, (1100, 600))]
            a, b = np.meshgrid(np.linspace(lim[0], lim[1], nu),
                               np.linspace(lim[2], lim[3], nv), indexing='ij')
            with np.errstate(all='ignore'):
                p = _hc_points(f, a, b)
            uv = np.stack((a, b), axis=-1)
            tri = np.concatenate((np.stack((p[:-1, :-1], p[1:, :-1], p[1:, 1:]), -2).reshape(-1, 3, 3),
                                  np.stack((p[:-1, :-1], p[1:, 1:], p[:-1, 1:]), -2).reshape(-1, 3, 3)))
            tuv = np.concatenate((np.stack((uv[:-1, :-1], uv[1:, :-1], uv[1:, 1:]), -2).reshape(-1, 3, 2),
                                  np.stack((uv[:-1, :-1], uv[1:, 1:], uv[:-1, 1:]), -2).reshape(-1, 3, 2)))
            for start in range(0, len(tri), 40000):
                t, tu = tri[start:start+40000], tuv[start:start+40000]
                keep = np.isfinite(t).all((1, 2)) & (t[:, :, 1].max(1) > 0)
                t, tu = t[keep], tu[keep]
                if not len(t):
                    continue
                xz = (t[:, :, (0, 2)]-origin)/step
                lo = np.maximum(np.ceil(xz.min(1)-1e-9).astype(int), 0)
                hi = np.minimum(np.floor(xz.max(1)+1e-9).astype(int), shape-1)
                span = np.maximum(hi-lo+1, 0)
                count = span[:, 0]*span[:, 1]
                keep = count > 0
                t, tu, lo, span, count = t[keep], tu[keep], lo[keep], span[keep], count[keep]
                if not len(t):
                    continue
                ids = np.repeat(np.arange(len(t)), count)
                off = np.arange(count.sum())-np.repeat(np.cumsum(count)-count, count)
                ix = lo[ids, 0]+off//span[ids, 1]
                iz = lo[ids, 1]+off % span[ids, 1]
                gx, gz = origin[0]+step*ix, origin[1]+step*iz
                aa, bb, cc = t[ids, 0], t[ids, 1], t[ids, 2]
                det = (bb[:, 2]-cc[:, 2])*(aa[:, 0]-cc[:, 0])+(cc[:, 0]-bb[:, 0])*(aa[:, 2]-cc[:, 2])
                with np.errstate(all='ignore'):
                    wa = ((bb[:, 2]-cc[:, 2])*(gx-cc[:, 0])+(cc[:, 0]-bb[:, 0])*(gz-cc[:, 2]))/det
                    wb = ((cc[:, 2]-aa[:, 2])*(gx-cc[:, 0])+(aa[:, 0]-cc[:, 0])*(gz-cc[:, 2]))/det
                wc = 1-wa-wb
                y = wa*aa[:, 1]+wb*bb[:, 1]+wc*cc[:, 1]
                keep = np.isfinite(y) & (y > 0) & (wa >= -1e-8) & (wb >= -1e-8) & (wc >= -1e-8)
                fi = ix[keep]*shape[1]+iz[keep]
                y, ids, wa, wb, wc = y[keep], ids[keep], wa[keep], wb[keep], wc[keep]
                if not len(fi):
                    continue
                np.maximum.at(height, fi, y)
                win = y == height[fi]
                wi = np.flatnonzero(win)
                _, pick = np.unique(fi[wi], return_index=True)
                wi = wi[pick]
                target = fi[wi]
                owner[target] = pi
                native[target] = (wa[wi, None]*tu[ids[wi], 0]+wb[wi, None]*tu[ids[wi], 1]
                                  +wc[wi, None]*tu[ids[wi], 2])
        self._cache = dict(origin=origin, shape=shape, height=height,
                           owner=owner, uv=native, step=step)

    def _cache_indices(self, query):
        c = self._cache
        position = (query-c['origin'])/c['step']
        safe = np.where(np.isfinite(position), position, -1e6)
        ij = np.floor(safe).astype(int)
        inside = (ij >= 0).all(1) & (ij < c['shape']-1).all(1) & np.isfinite(query).all(1)
        i = np.clip(ij[:, 0], 0, c['shape'][0]-2)
        j = np.clip(ij[:, 1], 0, c['shape'][1]-2)
        ids = np.stack((i*c['shape'][1]+j, (i+1)*c['shape'][1]+j,
                        i*c['shape'][1]+j+1, (i+1)*c['shape'][1]+j+1), axis=1)
        u, v = safe[:, 0]-ij[:, 0], safe[:, 1]-ij[:, 1]
        weights = np.stack(((1-u)*(1-v), u*(1-v), (1-u)*v, u*v), axis=1)
        return inside, ids, weights

    def _native_contact(self, query):
        """Refine all four cached winning branches; return actual y and jets."""
        self._build_cache()
        c = self._cache
        inside, ids, weights = self._cache_indices(query)
        owners = c['owner'][ids]
        out = np.full((len(query), 3), np.nan)
        # Adjacent cache nodes may belong to different overlapping surfaces.
        # Refining each represented branch prevents taking an arbitrary side
        # of a layer crossing when returning an exact depth/gradient.
        for pi in np.unique(owners[inside]):
            if pi < 0:
                continue
            match = (owners == pi) & inside[:, None]
            rows = np.flatnonzero(match.any(1))
            w = np.where(match[rows], np.maximum(weights[rows], 1e-12), 0)
            uv = np.sum(np.where(match[rows, :, None], c['uv'][ids[rows]], 0)*w[:, :, None], axis=1)/w.sum(1)[:, None]
            local_height = np.where(match[rows], c['height'][ids[rows]], -np.inf)
            outer_corner = np.argmax(local_height, axis=1)
            outer_uv = c['uv'][ids[rows, outer_corner]]
            for seed in (uv, outer_uv):
                y, gx, gz = self._invert(self.patches[int(pi)], query[rows], seed, True)
                take = np.isfinite(y) & (~np.isfinite(out[rows, 0]) | (y > out[rows, 0]))
                out[rows[take]] = np.c_[y, gx, gz][take]
        return out

    def depth(self, x, z):
        """Fast high-resolution upper-envelope depth (default pitch .004).

        Interior nodes use bilinear interpolation of the fine parameter-mesh
        upper envelope. Boundary nodes are refined on actual native surfaces.
        No mesh cells are dilated or filled across missing projections. Use
        ``depth_exact`` for diagnostic residual checks, ``gradient`` for native
        derivatives rather than derivatives of the interpolation grid.
        """
        if not self.cache_step:
            return self.depth_exact(x, z)
        self._build_cache()
        x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        query = np.c_[x.ravel(), z.ravel()]
        inside, ids, weights = self._cache_indices(query)
        values = self._cache['height'][ids]
        owners = self._cache['owner'][ids]
        finite_corners = np.isfinite(values).all(1)
        # A crescent's cut edge can have a real depth jump to the layer below.
        # Interpolating between different winners would bevel that jump by as
        # much as an entire strand thickness, not O(pitch**2). Refine those cells
        # on all represented native branches instead of smearing the envelope.
        with np.errstate(invalid='ignore'):
            rough = (owners != owners[:, :1]).any(1) | (np.ptp(values, axis=1) > 0.01)
        good = inside & finite_corners & ~rough
        y = np.full(len(query), np.nan)
        y[good] = np.sum(values[good]*weights[good], axis=1)
        edge = inside & ~good & (owners >= 0).any(1)
        if edge.any():
            y[edge] = self._native_contact(query[edge])[:, 0]
        outside = ~inside & np.isfinite(query).all(1)
        if outside.any():
            y[outside] = self.depth_exact(query[outside, 0], query[outside, 1])
        result = y.reshape(x.shape)
        return float(result) if not x.shape else result

    def gradient(self, x, z):
        """Native derivatives (dH/dx, dH/dz) of the winning actual patch.

        At a layer switch the upper envelope need not be differentiable; this
        returns the branch with greatest refined depth among neighbouring cache
        owners. Near a projection silhouette derivatives may be very large.
        Missing contacts return NaNs. This method does not use grid differences.
        """
        if not self.cache_step:
            raise ValueError('gradient requires the default contact seed cache')
        x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        out = self._native_contact(np.c_[x.ravel(), z.ravel()])
        gx, gz = out[:, 1].reshape(x.shape), out[:, 2].reshape(x.shape)
        if not x.shape:
            return float(gx), float(gz)
        return gx, gz

    def depth_smooth(self, x, z, sigma=0.035):
        """Gaussian-smoothed support approximation, NOT an actual contact jet.

        ``sigma`` is in model units. A separable, normalised Gaussian averages
        only finite cache nodes; zero padding and a finite four-sigma kernel
        prevent wrap-around or influence across arbitrarily large missing gaps.
        Original missing nodes remain missing: smoothing never grows the support
        domain or fills holes. Bilinear queries require four supported corners,
        return NaN outside that domain, and never invoke Newton at layer edges.

        Results are cached by sigma. They are continuous inside supported cells,
        but need not lie above actual hair, nor match its native value/gradient.
        Use actual root jets and independently validate visible intersections.
        Existing depth/depth_exact/gradient behaviour is intentionally unchanged.
        """
        sigma = float(sigma)
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError('sigma must be a finite nonnegative model distance')
        if not self.cache_step:
            raise ValueError('depth_smooth requires a positive cache_step')
        self._build_cache()
        maps = getattr(self, '_smooth_contact_maps', None)
        if maps is None:
            maps = self._smooth_contact_maps = {}
        if sigma not in maps:
            c = self._cache
            raw = c['height'].reshape(tuple(c['shape']))
            valid = np.isfinite(raw)
            if sigma == 0:
                smooth = np.where(valid, raw, np.nan)
            else:
                sd = sigma/c['step']
                radius = max(1, int(np.ceil(4*sd)))
                offsets = np.arange(-radius, radius+1, dtype=float)
                kernel = np.exp(-0.5*(offsets/sd)**2)
                kernel /= kernel.sum()

                def convolve(values, axis):
                    padding = [(0, 0), (0, 0)]
                    padding[axis] = (radius, radius)
                    padded = np.pad(values, padding, mode='constant', constant_values=0)
                    return np.apply_along_axis(
                        lambda row: np.convolve(row, kernel, mode='valid'), axis, padded)

                numerator = np.where(valid, raw, 0.0)
                denominator = valid.astype(float)
                for axis in (0, 1):
                    numerator = convolve(numerator, axis)
                    denominator = convolve(denominator, axis)
                smooth = np.full(raw.shape, np.nan)
                np.divide(numerator, denominator, out=smooth,
                          where=valid & (denominator > 0))
            maps[sigma] = smooth.ravel()
        x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        query = np.c_[x.ravel(), z.ravel()]
        inside, ids, weights = self._cache_indices(query)
        values = maps[sigma][ids]
        good = inside & np.isfinite(values).all(1)
        y = np.full(len(query), np.nan)
        y[good] = np.sum(values[good]*weights[good], axis=1)
        result = y.reshape(x.shape)
        return float(result) if not x.shape else result
