import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def make_excel():
    # Getting the current working directory
    cwd = os.getcwd()

    # Navigate up to the 'Classification' directory
    curr_path = os.path.dirname(cwd)
    classification_path = os.path.join(curr_path, 'Classification')

    # Creating excel folder
    excel_folder = os.path.join(classification_path, 'excel')
    os.makedirs(excel_folder, exist_ok=True)

    # Excel file path
    excel_path = os.path.join(excel_folder, 'image_dataset_info.xlsx')

    # Check if Excel file already exists
    if not os.path.exists(excel_path):
        # DataFrame to hold the information
        data = {'ID': [], 'Split': [], 'Target': []}

        # Iterating through each split and class
        for split in ['train', 'test', 'val']:
            for target in ['fire', 'nofire']:
                folder_path = os.path.join(classification_path, split, target)
                for image in os.listdir(folder_path):
                    data['ID'].append(image)
                    data['Split'].append(split)
                    data['Target'].append(target)

        # Creating a DataFrame
        df = pd.DataFrame(data)

        # Saving to Excel
        df.to_excel(excel_path, index=False)
        print("Excel file created at:", excel_path)
    else:
        print("Excel file already exists at:", excel_path)

    return excel_path
def data_vis():

    # Visualization of distribution by Target
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Target', data=data)
    plt.title('Distribution of Images by Target from train, test and valid')
    plt.xlabel('Target')
    plt.ylabel('Count')
    plt.show()

    plt.figure(figsize=(8, 6))
    train_df = data[data['Split'] == 'train']
    sns.countplot(x='Target', data=train_df)
    plt.title('Distribution of Images by Target from train')
    plt.xlabel('Target')
    plt.ylabel('Count')
    plt.show()

    # Visualization of distribution by Split
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Split', data=data)
    plt.title('Distribution of Images by Split')
    plt.xlabel('Split')
    plt.ylabel('Count')
    plt.show()

    # Stacked Bar Chart
    pivot_df = data.groupby(['Split', 'Target']).size().unstack()
    pivot_df.plot(kind='bar', stacked=True, figsize=(8, 6))
    plt.title('Distribution of Images by Target and Split')
    plt.xlabel('Split')
    plt.ylabel('Count')
    plt.show()

def data_cleaning():
    # 1. Check for Missing Values
    print("Missing values before cleaning:")
    print(data.isnull().sum())

    # Drop rows with any missing values
    data.dropna(inplace=True)

    # 2. Removing Duplicates
    data.drop_duplicates(inplace=True)

    # 3. Data Type Conversion
    # Convert 'ID' and 'Target' to string if they're not already
    data['ID'] = data['ID'].astype(str)
    data['Target'] = data['Target'].astype(str)

    # 4. Text Data Cleaning
    # Example: Ensuring all 'Target' entries are lowercase
    data['Target'] = data['Target'].str.lower()

    # 5. Data Consistency
    # Example: Standardize category names
    data['Target'].replace({'fire': 'Fire', 'nofire': 'NoFire'}, inplace=True)

    print("\nMissing values after cleaning:")
    print(data.isnull().sum())

    print("\nCleaned DataFrame:")
    print(data.head())




# Reading the Excel file
xl_path = make_excel()
data = pd.read_excel(xl_path)

# Now you can manipulate the DataFrame as needed
print(data.head())
print("\n shape of data: ", data.shape)

# Call functions to execute
data_vis()
data_cleaning()
