from fed_local_4 import *

device = "cuda:1"
# modalities = ['image', 'covid', 'ecg', 'clinicals']
modalities = ['ecg']
test_image_data_path = "/data/xiaochen/FedMFM/preprocessed_jsons/RSNA_test.json"
test_covid_data_path = "/data/xiaochen/FedMFM/preprocessed_jsons/covid_test.json"
test_ecg_data_path = "/data/xiaochen/FedMFM/preprocessed_jsons/ecg_test.json"
test_clinical_data_path = "/data/xiaochen/FedMFM/preprocessed_jsons/mortality_test.json"

test_data_paths = {
   'image': test_image_data_path,
    'covid': test_covid_data_path,
    'ecg': test_ecg_data_path,
    'clinicals': test_clinical_data_path
}

test_data_loaders = {
    'image': None,
    'covid': None,
    'ecg': None,
    'clinicals': None
}

# Parse test data for the specified modalities
for modality in modalities:
    if modality in test_data_paths:
        test_data_path = test_data_paths[modality]
        if test_data_path:
            if modality == 'image' or modality == 'covid':
                test_image_paths, test_image_labels = parse_data(test_data_path, modality)
                test_data_loaders[modality] = DataLoader(ImageDataset(test_image_paths, test_image_labels, AutoFeatureExtractor.from_pretrained('facebook/deit-tiny-distilled-patch16-224')), batch_size=32, shuffle=False)
            elif modality == 'ecg':
                test_ecg_paths, test_ecg_labels = parse_data(test_data_path, modality)
                test_data_loaders[modality] = DataLoader(ECGDataset(test_ecg_paths, test_ecg_labels), batch_size=32, shuffle=False)
            elif modality == 'clinicals':
                test_clinical_paths, test_clinical_labels = parse_data(test_data_path, modality)
                test_data_loaders[modality] = DataLoader(ClinicalDataset(test_clinical_paths, test_clinical_labels), batch_size=32, shuffle=False)

global_model = torch.load("/data/xiaochen/FedMFM/ckpt/ours_fedavg_ecg/5/local.pt")

if test_data_loaders['image']:
    print("Evaluating Image Model:")
    global_model.set_task('lung_opacity')  # Set the task as needed
    evaluate_image_model(global_model, test_data_loaders['image'], device)

if test_data_loaders['covid']:
    print("Evaluating COVID Detection Model:")
    global_model.set_task('covid_detection')  # Set the task as needed
    evaluate_image_model(global_model, test_data_loaders['covid'], device)

if test_data_loaders['ecg']:
    print("Evaluating ECG Model:")
    global_model.set_task('ecg_abnormal')  # Set the task as needed
    evaluate_ecg_model(global_model, test_data_loaders['ecg'], device)

if test_data_loaders['clinicals']:
    print("Evaluating Clinical Model:")
    global_model.set_task('mortality')  # Set the task as needed
    evaluate_clinical_model(global_model, test_data_loaders['clinicals'], device)
    
    
