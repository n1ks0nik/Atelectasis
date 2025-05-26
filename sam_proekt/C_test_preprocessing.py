from B_dicom_handler import preprocess_dicom
import matplotlib.pyplot as plt

img = preprocess_dicom(r"C:\Users\CYBER ARTEL\.cache\kagglehub\datasets\nih-chest-xrays\data\nih_custom_dataset\fake_dicom\pre_dicom\test_atelectasis.dcm")
plt.imshow(img, cmap="gray")
plt.title("Preprocessed DICOM")
plt.show()
