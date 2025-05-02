import pandas as pd
import os

def clean_hate_speech_data():
    """
    Cleans hate speech data by properly formatting the CSV
    and separating text comments from classification labels.
    """
    try:
        # Set up directory paths
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_dir, 'data')
        output_dir = os.path.join(project_dir, 'data')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Input and output file paths
        input_path = os.path.join(data_dir, 'hate_train.csv')
        output_path = os.path.join(output_dir, 'hate_train_cleaned.csv')
        
        print(f"Processing file: {input_path}")
        
        # Read the data - try multiple approaches
        try:
            # First attempt: read as-is (if it already has headers)
            df = pd.read_csv(input_path, sep=';', engine='python')
            
            # Check if we need to rename columns
            if len(df.columns) == 2 and 'comment' not in df.columns:
                df.columns = ['comment', 'isHate']
                
        except Exception as e:
            print(f"First attempt failed: {e}")
            
            # Second attempt: read without headers and assign them
            try:
                df = pd.read_csv(input_path, sep=';', header=None, engine='python')
                df.columns = ['comment', 'isHate']
            except Exception as e2:
                print(f"Second attempt failed: {e2}")
                
                # Third attempt: read as single column and split
                try:
                    df = pd.read_csv(input_path, header=None, engine='python')
                    df.columns = ['raw_data']
                    df[['comment', 'isHate']] = df['raw_data'].str.split(';', expand=True)
                    df = df[['comment', 'isHate']]
                except Exception as e3:
                    print(f"All automated attempts failed. Error: {e3}")
                    raise
        
        # Convert isHate to float
        df['isHate'] = df['isHate'].astype(float)
        
        # Display data preview
        print("\nData preview:")
        print(df.head())
        print(f"\nDataset shape: {df.shape}")
        
        # Save the cleaned data
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        
        return df
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

if __name__ == "__main__":
    clean_hate_speech_data()