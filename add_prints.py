import nbformat as nbf
from sys import argv
def insert_after_imports_and_line(filename, new_cell_1, new_cell_2, line):
    # Load the notebook.
    with open(filename) as f:
        nb = nbf.read(f, as_version=4)

    # Flag for detecting import cells.
    import_exists = False

    # Go through each cell.
    for idx, cell in enumerate(nb.cells):
        # If the cell has 'import' in it, mark it.
        if 'import' in cell['source']:
            import_exists = idx

        # If the cell contains the provided line, add the second cell after it.
        if line in cell['source']:
            nb.cells.insert(idx + 1, nbf.v4.new_code_cell(source=new_cell_2))

    # Add the first cell after the last import cell, or at the end.
    insert_at = import_exists + 1 if import_exists is not False else len(nb.cells)
    nb.cells.insert(insert_at, nbf.v4.new_code_cell(source=new_cell_1))

    # Write the notebook back to the file.
    with open(filename, 'w') as f:
        nbf.write(nb, f)

# Dummy usage example.A Python script that insert a new cell after the last import cell 
# filename = 'path_to_your_notebook.ipynb'
new_cell_1 = 'print("This is the first new cell.")'
new_cell_2 = 'print("This is the second new cell.")'
line = 'a_specific_line_of_code'
if __name__ == '__main__':
    insert_after_imports_and_line(argv[1], new_cell_1, new_cell_2, line)