# Nearest-neighbour distance (64x64 gray mean-abs-diff, min over horizontal flip)
# between md5-distinct images, to choose a near-duplicate threshold.
import os, csv
import numpy as np, torch
D = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(D, 'image_stats.csv'))))
th = np.load(os.path.join(D, 'thumbs.npz'))
first = {}
for i, r in enumerate(rows): first.setdefault(r['md5'], i)
u = np.array(sorted(first.values()))
src = np.array([rows[i]['source'] for i in u])

def nn(kind, q, ref, same):
    a = torch.from_numpy(th[kind][u].reshape(len(u), -1).astype(np.float32) / 255)
    m = torch.from_numpy(th[kind][u][:, :, ::-1].copy().reshape(len(u), -1).astype(np.float32) / 255)
    qi, ri = np.where(q)[0], np.where(ref)[0]
    best, arg = [], []
    for s in range(0, len(qi), 512):
        c = a[qi[s:s + 512]]
        d = torch.minimum(torch.cdist(c, a[ri], p=1), torch.cdist(c, m[ri], p=1)) / a.shape[1]
        if same:
            d[torch.from_numpy(qi[s:s + 512, None] == ri[None, :])] = 9
        v, j = d.min(1); best.append(v.numpy()); arg.append(ri[j.numpy()])
    return np.concatenate(best), np.concatenate(arg), qi

pool = np.isin(src, ['original', 'new'])
edges = [0, .005, .01, .015, .02, .025, .03, .04, .05, .07, .1, 1]
out = {}
for kind in ['raw', 'crop']:
    for name, q, ref, same in [('pool->pool', pool, pool, True), ('testing->pool', src == 'testing', pool, False),
                               ('external->pool', src == 'external', pool, False), ('external->testing', src == 'external', src == 'testing', False)]:
        b, a, qi = nn(kind, q, ref, same)
        out[(kind, name)] = (b, a, qi)
        print(f"{kind:4s} {name:18s} n={len(b):5d} NN-dist hist:", np.histogram(b, edges)[0].tolist())
print("bins:", edges)
np.savez(os.path.join(D, 'nn.npz'), **{f"{k}|{n}|{w}": v[i] for (k, n), v in out.items() for i, w in enumerate(['d', 'arg', 'q'])}, u=u)
