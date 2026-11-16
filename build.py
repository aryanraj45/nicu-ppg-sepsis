"""Convert notebook_src.py (# %% cell markers) into nicu_ppg_sepsis.ipynb."""
import re, nbformat as nbf
src = open('notebook_src.py').read()
cells = []
for chunk in re.split(r'^# %%', src, flags=re.M)[1:]:
    header, _, body = chunk.partition('\n')
    if header.strip() == '[markdown]':
        text = '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.strip().splitlines())
        cells.append(nbf.v4.new_markdown_cell(text))
    else:
        cells.append(nbf.v4.new_code_cell(body.strip()))
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, 'nicu_ppg_sepsis.ipynb')
print(len(cells), 'cells')
