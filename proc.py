#%%
from glob import glob
modules = glob("lessons/*")

#%% 
for module in sorted(modules):
    print('\t\tpart:', '"' + ' '.join(module.split('/')[1].split('_')) + '"')
    print("\t\tchapters:")
    for notebook in sorted(glob(f"{module}/*.ipynb")):
        print(f"\t\t\t-{notebook}")
# %%
