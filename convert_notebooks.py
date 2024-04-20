from pathlib import Path
import shutil
def main():
    for notebook_fname in Path('notebooks').glob('**/*.ipynb'):
        two_above  = notebook_fname.parent.parent
        if (two_above / 'src').exists() and (two_above / 'input').exists():
            # for fname in two_above.glob('*'):
            shutil.move(notebook_fname, two_above)
            
if __name__ == '__main__':
    main()