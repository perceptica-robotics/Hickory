import numpy as np
import clipperpy
from scipy.special import beta 
from roman.align.object_registration import ObjectRegistration
from roman.object.object import Object

def get_exact_sq_volume(params):
    """calculates the exact volume of a single superquadric given its parameters using the analytical formula."""
    # params: [e1, e2, ax, ay, az]
    e1, e2 = max(params[0], 1e-6), max(params[1], 1e-6)
    ax, ay, az = params[2], params[3], params[4]
    
    term_scale = 2.0 * ax * ay * az
    term_shape = e1 * e2
    term_beta = beta(e1/2 + 1, e1) * beta(e2/2, e2/2)
    
    return term_scale * term_shape * term_beta

def get_volume_weights(sqs_A, sqs_B, num_items):
    if num_items <= 0: return np.array([])

    volumes = []
    for i in range(num_items):
        vol_A = get_exact_sq_volume(sqs_A[i])
        vol_B = get_exact_sq_volume(sqs_B[i])
        avg_vol = (vol_A + vol_B) / 2.0
        volumes.append(avg_vol)
    
    volumes = np.array(volumes, dtype=float)
    total = volumes.sum()
    if total == 0: return np.ones(num_items) / num_items
    return volumes / total

class SQObject(Object):
    def __init__(self, centroid, sq_params, semantic_feature, id=0, coarse_shape_attrs=None):
        super().__init__(centroid=np.array(centroid), id=id) 
        self.sq_params = np.atleast_2d(np.array(sq_params))
        self.coarse_shape_attrs = None if coarse_shape_attrs is None else np.asarray(coarse_shape_attrs, dtype=float).reshape(-1)
        
        if isinstance(semantic_feature, dict):
            self.semantic_feature = semantic_feature['mean'].flatten()
        else:
            self.semantic_feature = semantic_feature.flatten()

    @property
    def pos_flat(self):
        return self.centroid.flatten()

class SQRegistration(ObjectRegistration):
    def __init__(self, geom_sigma=0.5, shape_sigma=1.0, shape_mode="sq", corr_weight_power=1.0, min_corr_weight=1e-6):
        super().__init__(dim=3)
        self.geom_sigma = geom_sigma
        self.shape_sigma = shape_sigma
        self.shape_mode = str(shape_mode).lower()
        self.corr_weight_power = float(corr_weight_power)
        self.min_corr_weight = float(min_corr_weight)
        if self.shape_mode not in ("sq", "coarse", "none"):
            raise ValueError(f"Unsupported shape_mode: {shape_mode}. Use one of: sq, coarse, none.")

    def _compute_sq_shape_score(self, sq_A: SQObject, sq_B: SQObject):
        params_A = sq_A.sq_params
        params_B = sq_B.sq_params
        
        num_to_compare = int(1.0 * min(len(params_A), len(params_B)))
        if num_to_compare < 1:
            num_to_compare = 1
            
        weights = get_volume_weights(params_A, params_B, num_to_compare)
        sq_total_score = 0.0
        for i in range(num_to_compare):
            pA = params_A[i]
            pB = params_B[i]

            diff_e = abs(pA[0] - pB[0]) + abs(pA[1] - pB[1])
            s_shape = np.exp(-diff_e / self.shape_sigma)

            dims_A = pA[2:5]
            dims_B = pB[2:5] + 1e-6
            ratio = np.minimum(dims_A, dims_B) / np.maximum(dims_A, dims_B)
            s_scale = np.mean(ratio)

            sq_total_score += (s_shape * s_scale) * weights[i]
        return float(sq_total_score)

    def _compute_coarse_shape_score(self, sq_A: SQObject, sq_B: SQObject):
        a = sq_A.coarse_shape_attrs
        b = sq_B.coarse_shape_attrs
        if a is None or b is None:
            return 1.0
        if a.size == 0 or b.size == 0:
            return 1.0
        n = min(a.size, b.size)
        a = a[:n]
        b = b[:n]
        eps = 1e-8
        a_safe = np.maximum(a, eps)
        b_safe = np.maximum(b, eps)
        ratio = np.minimum(a_safe, b_safe) / np.maximum(a_safe, b_safe)
        return float(np.mean(np.clip(ratio, 0.0, 1.0)))

    def _compute_unary_details(self, sq_A: SQObject, sq_B: SQObject):
        if self.shape_mode == "sq":
            s_shape_final = self._compute_sq_shape_score(sq_A, sq_B)
        elif self.shape_mode == "coarse":
            s_shape_final = self._compute_coarse_shape_score(sq_A, sq_B)
        else:
            s_shape_final = 1.0

        # 4. Semantic Score
        fA, fB = sq_A.semantic_feature, sq_B.semantic_feature
        s_sem = max(0.0, np.dot(fA, fB) / (np.linalg.norm(fA)*np.linalg.norm(fB) + 1e-8))

        # 5. Final Hybrid Score
        if self.shape_mode == "none":
            s_total = s_sem
        else:
            s_total = np.sqrt(s_shape_final * s_sem)
        
        return s_shape_final, s_sem, s_total

    def _compute_unary_score(self, sq_A, sq_B):
        _, _, total = self._compute_unary_details(sq_A, sq_B)
        return total

    def _compute_pairwise_score(self, dist_A, dist_B):
        return np.exp(-abs(dist_A - dist_B) / self.geom_sigma)

    def register_sq_maps(self, mapA: list[SQObject], mapB: list[SQObject], return_corrs: bool = False):
        n1, n2 = len(mapA), len(mapB)
        associations = [(i, j) for i in range(n1) for j in range(n2)]
        num_nodes = len(associations)
        
        if num_nodes == 0: return np.eye(4)

        M = np.zeros((num_nodes, num_nodes))
        unary_scores = np.zeros(num_nodes)

        for k, (u, v) in enumerate(associations):
            unary_scores[k] = self._compute_unary_score(mapA[u], mapB[v])
            M[k, k] = np.cbrt(unary_scores[k] * unary_scores[k])

        for k in range(num_nodes):
            for l in range(k + 1, num_nodes):
                u, v = associations[k]
                p, q = associations[l]
                if u == p or v == q: continue 
                if unary_scores[k] < 0.1 or unary_scores[l] < 0.1: continue

                dA = np.linalg.norm(mapA[u].center - mapA[p].center)
                dB = np.linalg.norm(mapB[v].center - mapB[q].center)
                s_geom = self._compute_pairwise_score(dA, dB)
                combined_score = np.cbrt(s_geom * unary_scores[k] * unary_scores[l])
                M[k, l] = M[l, k] = combined_score

        invariant = clipperpy.invariants.PairwiseInvariant()
        params = clipperpy.Params()
        clipper = clipperpy.CLIPPER(invariant, params)
        clipper.set_matrix_data(M, np.ones_like(M))
        clipper.solve()
        
        solution = clipper.get_solution()
        nodes = solution.nodes
        
        scored_corrs = []
        print("\n" + "="*80)
        print(f"CLIPPER Analysis (shape_mode={self.shape_mode}): Found {len(nodes)} associations")
        print("-" * 80)
        for node_idx in nodes:
            u, v = associations[node_idx]
            s_shape, s_sem, s_total = self._compute_unary_details(mapA[u], mapB[v])
            scored_corrs.append((u, v, s_shape, s_sem, s_total))
            print(f"Obj {u} <-> Obj {v} | Shape: {s_shape:.4f} | Sem: {s_sem:.4f} | Total: {s_total:.4f}")
        print("="*80)

        scored_corrs.sort(key=lambda item: item[4], reverse=True)
        used_a = set()
        used_b = set()
        corrs = []
        corr_weights = []
        for u, v, _, _, s_total in scored_corrs:
            if u in used_a or v in used_b:
                continue
            used_a.add(u)
            used_b.add(v)
            corrs.append((u, v))
            w = max(float(s_total), self.min_corr_weight) ** self.corr_weight_power
            corr_weights.append(w)

        if len(corrs) < 3:
            print("Warning: < 3 matches found.")
            T = np.eye(4)
        else:
            T = self.T_align(mapA, mapB, np.array(corrs), weights=np.array(corr_weights, dtype=float))

        if return_corrs:
            return T, corrs
        return T
