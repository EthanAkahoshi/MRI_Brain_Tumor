# Group sizes and Testing/external contamination at candidate thresholds
# (union-find over md5-distinct images of pool + Testing + external, raw thumbs).
import os, csv
import numpy as np, torch
D = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(D, 'image_stats.csv'))))
th = np.load(os.path.join(D, 'thumbs.npz'))['raw']
first = {}
for i, r in enumerate(rows): first.setdefault(r['md5'], i)
u = np.array(sorted(first.values())); src = np.array([rows[i]['source'] for i in u])
a = torch.from_numpy(th[u].reshape(len(u), -1).astype(np.float32) / 255)
m = torch.from_numpy(th[u][:, :, ::-1].copy().reshape(len(u), -1).astype(np.float32) / 255)
pairs = []
for s in range(0, len(u), 512):
    d = torch.minimum(torch.cdist(a[s:s+512], a, p=1), torch.cdist(a[s:s+512], m, p=1)) / a.shape[1]
    i, j = (d < 0.04).nonzero(as_tuple=True)
    keep = (i + s) < j
    pairs.append(np.stack([(i + s)[keep].numpy(), j[keep].numpy(), d[i, j][keep].numpy()], 1))
pairs = np.concatenate(pairs)
pool = np.isin(src, ['original', 'new']); held = ~pool
# md5-level: pool md5s that equal a Testing/external md5 are already "held" because first index wins? no:
md5_held = {rows[i]['md5'] for i in range(len(rows)) if rows[i]['source'] in ('testing', 'external')}
pool_md5_hit = np.array([rows[i]['md5'] in md5_held for i in u]) & pool
for t in [0.015, 0.02, 0.025, 0.03, 0.035]:
    p = pairs[pairs[:, 2] < t]
    parent = np.arange(len(u))
    def f(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j, _ in p: parent[f(int(i))] = f(int(j))
    g = np.array([f(x) for x in range(len(u))])
    direct = np.zeros(len(u), bool)
    for i, j, _ in p:
        i, j = int(i), int(j)
        if pool[i] and held[j]: direct[i] = True
        if pool[j] and held[i]: direct[j] = True
    held_groups = set(g[held]) | set(g[pool_md5_hit])
    in_held_group = np.isin(g, list(held_groups)) & pool
    gp = g[pool & ~in_held_group]
    sizes = np.bincount(np.unique(gp, return_inverse=True)[1])
    print(f"t={t}: pool distinct {pool.sum()}, md5-hit {pool_md5_hit.sum()}, direct-near-held {(direct & ~pool_md5_hit).sum()}, "
          f"removed if whole group touches held {in_held_group.sum()} -> remaining {len(gp)} in {len(sizes)} groups, "
          f"largest {sizes.max()}, groups>=20: {(sizes >= 20).sum()}")
