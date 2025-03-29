import numpy as np
import random
from random import sample
import scipy.sparse as sp
from scipy.sparse import lil_matrix, csc_matrix

def inject_spike(Nall, M, N, Dspike, C):
    Nstart, i = Nall, Nall
    injectEs = list()
    injectUs, injectVs = range(Nall, Nall + M, 1), range(Nall, Nall + N, 1)
    for m in range(M):
        # standard normal distribution
        v1, v2, w = 0.0, 0.0, 2.0
        while w > 1.0:
            v1 = random.random() * 2.0 - 1.0
            v2 = random.random() * 2.0 - 1.0
            w = v1 * v1 + v2 * v2
        outd = int(Dspike + v1 * np.sqrt(-2.0 * np.log(w) / w))
        if outd < 0:  outd = Dspike
        outdC = int(outd * C)
        outdN = outd - outdC
        Ns, Cs = set(), set()
        for d in range(outdN):
            Ns.add(Nstart + M + random.randint(N))
        for d in range(outdC):
            Cs.add(random.randint(Nall))

        for j in Ns:
            injectEs.append([i, j])
        for j in Cs:
            injectEs.append([i, j])
        i += 1
    return len(injectEs), injectEs, injectUs, injectVs

def inject_gap(Nall, M, N, D0, Dgap, C):
    injectEs = list()
    injectUs, injectVs = range(Nall, Nall + M, 1), range(Nall, Nall + N, 1)
    Nstart, i = Nall, Nall
    Md = int(1.0 * M / (Dgap - D0 + 1))
    for outd in range(D0, Dgap, 1):
        for m in range(Md):
            outdC = int(outd * C)
            outdN = outd - outdC
            Ns, Cs = set(), set()
            for d in range(outdN):
                Ns.add(Nstart + M + random.randint(N))
            for d in range(outdC):
                Cs.add(random.randint(Nall))
            for j in Ns:
                injectEs.append([i, j])
            for j in Cs:
                injectEs.append([i, j])
            i += 1

    return len(injectEs), injectEs, injectUs, injectVs


def genEvenDenseBlock(A, B, p):
    m = []
    for i in range(A):
        a = np.random.binomial(1, p, B)
        m.append(a)
    return np.array(m)

def genHyperbolaDenseBlock(A, B, alpha, tau):
    'this is from hyperbolic paper: i^\alpha * j^\alpha > \tau'
    m = np.empty([A, B], dtype=int)
    for i in range(A):
        for j in range(B):
            if (i+1)**alpha * (j+1)**alpha > tau:
                m[i,j] = 1
            else:
                m[i,j] = 0
    return m

def genDiHyperRectBlocks(A1, B1, A2, B2, alpha=-0.5, tau=None, p=1):
    if tau is None:
        tau = A1**alpha * B1**alpha
    m1 = genEvenDenseBlock(A1, B1, p=p)
    m2 = genHyperbolaDenseBlock(A2, B2, alpha, tau)
    M = sp.linalg.block_diag(m1, m2)
    return M

def addnosie(M, A, B, p, black=True, A0=0, B0=0):
    v = 1 if black else 0
    for i in range(A-A0):
        a = np.random.binomial(1, p, B-B0)
        for j in a.nonzero()[0]:
            M[A0+i,B0+j]=v
    return M


def injectCliqueCamo(M, m0, n0, p, testIdx):
    (m,n) = M.shape
    M2 = M.copy().tolil()

    colSum = np.squeeze(M2.sum(axis = 0).A)
    colSumPart = colSum[n0:n]
    colSumPartPro = np.int_(colSumPart)
    colIdx = np.arange(n0, n, 1)
    population = np.repeat(colIdx, colSumPartPro, axis = 0)

    for i in range(m0):
        # inject clique
        for j in range(n0):
            if random.random() < p:
                M2[i,j] = 1
        # inject camo
        if testIdx == 1:
            thres = p * n0 / (n - n0)
            for j in range(n0, n):
                if random.random() < thres:
                    M2[i,j] = 1
        if testIdx == 2:
            thres = 2 * p * n0 / (n - n0)
            for j in range(n0, n):
                if random.random() < thres:
                    M2[i,j] = 1
        # biased camo
        if testIdx == 3:
            colRplmt = random.sample(population, int(n0 * p))
            M2[i,colRplmt] = 1

    return M2.tocsc()


def generateProps(rates, times, k, s, t0, tsdiffcands, tsp):

    if len(rates) > 0:
        rs = np.random.choice([4, 4.5], size=s)
        if k in rates:
            for r in rs:
                rates[k].append(r)
        else:
            rates[k] = list(rs)
    if len(times) > 0:
        ts = np.random.choice(tsdiffcands, size=s, p=tsp) + t0
        if k in times:
            for t in ts:
                times[k].append(t)
        else:
            times[k] = list(ts)
    return

def injectCliqueCamo(M, m0, n0, p, testIdx):
    (m,n) = M.shape
    M2 = M.copy().tolil()

    colSum = np.squeeze(M2.sum(axis = 0).A)
    colSumPart = colSum[n0:n]
    colSumPartPro = np.int_(colSumPart)
    colIdx = np.arange(n0, n, 1)
    population = np.repeat(colIdx, colSumPartPro, axis = 0)

    for i in range(m0):
        # inject clique
        for j in range(n0):
            if random.random() < p:
                M2[i,j] = 1
        # inject camo
        if testIdx == 1:
            thres = p * n0 / (n - n0)
            for j in range(n0, n):
                if random.random() < thres:
                    M2[i,j] = 1
        if testIdx == 2:
            thres = 2 * p * n0 / (n - n0)
            for j in range(n0, n):
                if random.random() < thres:
                    M2[i,j] = 1
        # biased camo
        if testIdx == 3:
            colRplmt = random.sample(population, int(n0 * p))
            M2[i,colRplmt] = 1

    return M2.tocsc()

def inject_clique_camo(M, m0, n0, p, testIdx):
    (m, n) = M.shape
    injectEs = list()
    injectUs, injectVs = np.arange(m0), np.arange(n0)

    if testIdx in [3, 4]:  # popular biased camouflage
        colSum = np.squeeze(M.sum(axis = 0).A)
        colSumPart = colSum[n0:n]
        colSumPartPro = np.int_(colSumPart)
        colIdx = np.arange(n0, n, 1)
        population = np.repeat(colIdx, colSumPartPro, axis = 0)

    for i in range(m0):
        # inject clique
        for j in range(n0):
            if np.random.random() < p:
                injectEs.append([i,j])

        if testIdx == 0:
            continue
        # inject random camo
        if testIdx == 1:
            thres = p * n0 / (n - n0)
            for j in range(n0, n):
                if np.random.random() < thres:
                    injectEs.append([i,j])
        if testIdx == 2:
            thres = 2 * p * n0 / (n - n0)
            for j in range(n0, n):
                if np.random.random() < thres:
                    injectEs.append([i,j])
        # biased camo
        if testIdx == 3:
            colRplmt = sample(population, int(n0 * p))
            for j in colRplmt:
                injectEs.append([i,j])
        if testIdx == 4:
            colRplmt = sample(population, int(2* n0 * p))
            for j in colRplmt:
                injectEs.append([i,j])

    return len(injectEs), injectEs, injectUs, injectVs


# inject appended m0 by n0 camouflages to background graph M  (cpy & paste patterns)
# add new nodes and edges
def inject_append_cps_camo(M, m0, n0, p, camos):
    (m, n) = M.shape
    injectEs = list()
    injectUs, injectVs = np.arange(m0) + m,  np.arange(n0) + n

    col_sum = np.squeeze(M.sum(axis = 0).A)
    col_sumpro = np.int_(col_sum)
    col_idx = np.arange(n)
    pops = np.repeat(col_idx, col_sumpro, axis = 0)

    # inject dependent block
    for i in injectUs:
        for j in injectVs:
            pe = random.random()
            if pe < p:  injectEs.append([i, j])

        if camos == 0:  pass    # no camo
        if camos == 1:
            # random camo
            thres = p * n0 / (n - n0)
            for j in range(n):
                pe = random.random()
                if pe < thres:  injectEs.append([i, j])
        if camos == 2:
            # popular biased camo
            col_pops = random.sample(pops, int(n0 * p))
            for j in col_pops:    injectEs.append([i, j])

    return len(injectEs), injectEs, injectUs, injectVs

# pick nodes in original graph and add new edges
def inject_promot_camo(M, ms, ns, p, camos):
    (m, n) = M.shape
    M2 = M.copy()
    m0, n0 = len(ms), len(ns)

    injectEs = list()
    injectUs, injectVs = np.asarray(ms, dtype=int), np.asarray(ns, dtype=int)

    if camos in [3, 4, 5]:
        col_sum = np.squeeze(M2.sum(axis = 0).A)
        col_idx = np.setdiff1d(np.arange(n, dtype=int), injectVs)
        col_sumpart = col_sum[col_idx]
        pops = np.repeat(col_idx, np.int_(col_sumpart), axis = 0)

    for i in injectUs:
        # inject clique
        for j in injectVs:
            if random.random() < p and M2[i, j] == 0:
                M2[i, j] = 1
                injectEs.append([i, j])

        if camos == 0:
            continue
        if camos == 1:
            # random camo
            thres = p * n0 / (n - n0)
            for j in range(n):
                pe = random.random()
                if pe < thres and M2[i, j] == 0:
                    M2[i, j] = 1
                    injectEs.append([i, j])
        if camos == 2:
            # random camo
            thres = 2 * p * n0 / (n - n0)
            for j in range(n):
                pe = random.random()
                if pe < thres and M2[i, j] == 0:
                    M2[i, j] = 1
                    injectEs.append([i, j])
        if camos in [3, 4, 5]:
            # popular biased camo
            n0p = 0
            if camos == 4: n0p = 0.5 * n0 *p
            elif camos == 3: n0p = n0 * p
            elif camos == 5: n0p = 2 * n0 * p

            col_pops = random.sample(pops, int(n0p))
            for j in col_pops:
                if M2[i, j] == 0:
                    M2[i, j] = 1
                    injectEs.append([i, j])

    return M2, injectEs, injectUs, injectVs

def inject_fraud_const_objs(M, ms, ns, p, testIdx):
    M2 = M.copy()

    injectEs = list()
    injectUs = np.asarray(ms, dtype=int)
    injectVs = np.asarray(ns, dtype=int)

    if testIdx == 0:
        M2[ms, :][:, ns] = 0
        nmps = int(p * len(ms))
        for j in injectVs:
            for i in random.sample(injectUs, nmps):
                if M2[i, j] == 0:
                    M2[i, j] = 1
                    injectEs.append([i, j])
    elif testIdx == 1:
        for i in injectUs:
            for j in injectVs:
                if random.random() < p and M2[i, j] == 0:
                    M2[i, j] = 1
                    injectEs.append([i, j])

    return M2, injectEs, injectUs, injectVs

def injected_camos(M, ms, ns, p, camos):
    (m, n) = M.shape
    M1 = M.copy()
    m0, n0 = len(ms), len(ns)

    otherns = np.setdiff1d(np.arange(n, dtype=int), ns)

    for i in ms:
        if camos == 1:  # random camo
            thres = p * n0 / (n - n0)
            for j in otherns:
                if random.random() < thres:
                    M1[i, j] = 1
        if camos in [3, 4, 5]:  # biased camo
            col_sum = np.squeeze(M.sum(axis = 0).A)
            col_sumpart = col_sum[otherns]
            pops = np.repeat(otherns, np.int_(col_sumpart), axis = 0)

            n0p = n0 * p
            if camos == 3:  n0p *= 0.25
            if camos == 4:  n0p *= 0.5
            col_pops = random.sample(pops, int(n0p))
            for j in col_pops:
                M1[i, j] = 1
    return M1

def inject_jelly_attack(M, ms, ns, pns, p1, p2):
    (m, n) = M.shape
    M2 = M.copy()
    m0, n0, n1 = len(ms), len(ns), len(pns)

    injectEs = list()
    # col_idx = pns
    # col_sum = np.squeeze(M2[:, pns].sum(axis = 0).A)
    # pops = np.repeat(col_idx, np.int_(col_sum), axis = 0)

    for i in ms:
        for j in ns:
            if random.random() < p1 and M2[i, j] == 0:
                M2[i, j] = 1
                injectEs.append([i, j])

        for j in pns:
            if random.random() < p2 and M2[i, j] == 0:
                M2[i, j] = 1
                injectEs.append([i, j])

    return M2, injectEs, ms, ns