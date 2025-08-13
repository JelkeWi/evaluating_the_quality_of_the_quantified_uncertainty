import logging, os
import pandas as pd

def concat_excel_files(directory, keyword, keep_originals=False):
    # Initialize an empty list to store concatenated DataFrames
    dfs = []
    
    # Iterate over all files in the specified directory
    logging.info(f'Searching for files in: {directory}')
    for filename in os.listdir(directory):
        if filename.endswith('.xlsx') and keyword in filename:
            try:
                # Attempt to read the Excel file
                df = pd.read_excel(os.path.join(directory, filename), engine='openpyxl', index_col=0)
                dfs.append(df)
            except Exception as e:
                logging.error(f"Error reading {filename}: {e}")
    
    # Concatenate the DataFrames and save the result to a new Excel file
    try:
        concatenated_df = pd.concat(dfs, ignore_index=True)
        concatenated_df = concatenated_df.reset_index(drop=True)
        # Save the concatenated file in the same folder as the original files
        concatenated_df.to_excel(os.path.join(directory, keyword + ".xlsx"), index=True)
        
        if keep_originals:
            logging.info(f"Files with '{keyword}' in their names have been concatenated successfully.")
        else:
            logging.info("Files with '" + keyword + "' in their names have been concatenated. Deleting original files...")
            
            # Delete the original files
            for filename in os.listdir(directory):
                if filename.endswith('.xlsx') and keyword not in filename:
                    os.remove(os.path.join(directory, filename))
        
        logging.info("Concatenation complete.")
    
    except Exception as e:
        logging.error(f"Error concatenating files: {e}")
    return None