"""Parse the printed report tables of RREFL_R3_final.ipynb into JSON (no numbers typed by hand)."""
import json, re, sys
nb = json.load(open(sys.argv[1]))
def cell_text(i):
    return ''.join(''.join(o.get('text', '')) for o in nb['cells'][i].get('outputs', []))
CELL = r'(-?\d+\.\d+) \[(-?\d+\.\d+), (-?\d+\.\d+)\] \(n=(\d+)\)'
def trip(s):
    m = re.fullmatch(CELL, s.strip())
    return None if not m else dict(med=float(m[1]), lo=float(m[2]), hi=float(m[3]), n=int(m[4]))
out = {}
# ---- Table 1 (cell 40): rows = metrics, columns = arms
t1 = cell_text(40).splitlines()
arms = t1[1].split()[1:]
T1 = {a: {} for a in arms}
for line in t1[3:]:
    cells = re.findall(CELL + r'|n/a', line)
    vals = re.findall(r'(-?\d+\.\d+ \[-?\d+\.\d+, -?\d+\.\d+\] \(n=\d+\)|n/a)', line)
    if len(vals) != len(arms):
        continue
    name = line[:line.find(vals[0])].strip() if vals[0] != 'n/a' else line.split('  ')[0].strip()
    for a, v in zip(arms, vals):
        T1[a][name] = trip(v) if v != 'n/a' else None
out['T1'] = T1
# ---- generic row parser for tables 2/4/5 (DF40 blocks only)
def block(text, start, stop=None):
    s = text.index(start); e = text.index(stop, s + 1) if stop and stop in text[s + 1:] else len(text)
    return text[s:e].splitlines()
def rows(lines, keys):
    R = []
    for line in lines:
        vals = re.findall(r'-?\d+\.\d+ \[-?\d+\.\d+, -?\d+\.\d+\] \(n=\d+\)', line)
        if not vals: continue
        head = line[:line.find(vals[0])].split()
        fired = re.search(r'(\d+)/(\d+)\s*$', line)
        R.append(dict(head=head, vals=[trip(v) for v in vals], fired=(int(fired[1]) if fired else None)))
    return R
c43 = cell_text(43)
k = block(c43, 'TABLE 4 [DF40]', 'TABLE 5 [DF40]')
out['T4_DF40'] = [dict(arm=r['head'][0], method=r['head'][1], K=int(r['head'][2]),
                       bal=r['vals'][0], auc=r['vals'][1], seen=r['vals'][2], fired=r['fired']) for r in rows(k[2:], None)]
s5 = block(c43, 'TABLE 5 [DF40]', 'TABLE 4 [DeepSpeak]')
T5 = []
for r in rows(s5[2:], None):
    arm, lab = r['head'][0], ' '.join(r['head'][1:])
    m = re.match(r'(QA|FTR|FT)(?: (?:lr x|eps=)([\d.]+))?$', lab)
    meth, step = m[1], float(m[2]) if m[2] else (0.04 if m[1] == 'QA' else 1.0)
    T5.append(dict(arm=arm, method=meth, step=step, bal=r['vals'][0], auc=r['vals'][1], seen=r['vals'][2]))
out['T5_DF40'] = T5
c41 = cell_text(41)
t2 = block(c41, 'TABLE 2 [DF40]', '(all budgets')
T2 = []
for r in rows(t2[2:], None):
    h = r['head']
    if h[1] == 'zero-shot':
        T2.append(dict(arm=h[0], label='zero-shot', mode=None, auc=r['vals'][0]))
    else:
        T2.append(dict(arm=h[0], label=h[1], mode=h[2], auc=r['vals'][0], seen=r['vals'][4], fired=r['fired']))
out['T2_DF40'] = T2
# ---- rollback fired vs step (cell 53)
c53 = cell_text(53).splitlines()
F = {}
cur = None
for line in c53:
    m = re.match(r'^(\S+)?\s+(QA|FTR|FT)\s+((?:\d+/9\s*){5})$', line)
    if m:
        if m[1]: cur = m[1]
        F.setdefault(cur, {})[m[2]] = [int(x.split('/')[0]) for x in m[3].split()]
out['fired_vs_step'] = dict(steps=[0.04, 0.12, 0.35, 0.5, 1.0], counts=F)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
print('arms', arms); print('T1 metrics', list(T1['RR_uniform'].keys()))
print('T4 rows', len(out['T4_DF40']), 'T5 rows', len(T5), 'T2 rows', len(T2)); print(json.dumps(F))
