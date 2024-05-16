# Experimental methodology

- Sort notebooks by longest-running cell
- For each notebook, start by selecting the longest running cell and change the selection using the following criteria:
    - If the cell is an import or involves loading data, skip to the next longest-running cell
    - If at any point the cell runs for less than 3 seconds, skip to the next notebook
- If the cell involves a function call, move the function into the cell. 
- For the selected cell, run the Snappy magic to retrieve an annotated function to pass to Snappy. 
