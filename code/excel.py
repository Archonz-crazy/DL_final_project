import os

cwd = os.getcwd()
curr_path = os.path.dirname(cwd)
classification_path = os.path.join(curr_path, 'Classification')
train_dir = os.path.join(classification_path, 'train')
valid_dir = os.path.join(classification_path, 'valid')
test_dir = os.path.join(classification_path, 'test')
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
        for split in ['train', 'test', 'valid']:
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

xl_path = make_excel()