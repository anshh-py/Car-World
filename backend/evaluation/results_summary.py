import json, glob, os

files = sorted(glob.glob('results/*.json'))
for f in files:
    d = json.load(open(f, encoding='utf-8'))
    n = len(d.get('results', []))
    if n == 0:
        print(os.path.basename(f), 'EMPTY')
        continue
    outs = [(r['id'], r.get('provider_used'), r.get('outcome'), r.get('provider_requested')) for r in d['results']]
    gq = [o for o in outs if o[1] == 'groq']
    print(os.path.basename(f), 'total=%d groq=%d prov=%s' % (n, len(gq), d.get('providers_requested')))
    for i, p, o, pr in outs:
        print('   %s %s/%s -> %s' % (i, p, o, pr))
