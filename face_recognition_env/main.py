import cv2
import dlib
import numpy as np
import os
import matplotlib.pyplot as plt

# Path ke folder gambar
KNOWN_FACES_DIR = "images/known_faces"
TEST_IMAGES_DIR = "images/test_images"
RESULTS_DIR = "images/results"

# Pastikan folder results ada
os.makedirs(RESULTS_DIR, exist_ok=True)

# Inisialisasi model Dlib
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Fungsi untuk memuat wajah yang dikenali
def load_known_faces():
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Deteksi wajah
            faces = detector(rgb_image)
            if len(faces) > 0:
                shape = sp(rgb_image, faces[0])
                encoding = np.array(facerec.compute_face_descriptor(rgb_image, shape))
                known_face_encodings.append(encoding)
                known_face_names.append(name)
            else:
                print(f"Tidak ada wajah yang ditemukan di {filename}")

    return known_face_encodings, known_face_names

# Fungsi untuk mengonversi gambar ke grayscale
def convert_to_grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

# Fungsi untuk mengekstrak ROI berukuran 10x10 piksel dari pusat wajah
def extract_small_roi(image, roi, size=(10, 10)):
    """
    roi: Tuple (x, y, width, height) yang mendefinisikan region of interest.
    size: Ukuran ROI yang diinginkan (default: 10x10).
    """
    x, y, w, h = roi
    center_x = x + w // 2
    center_y = y + h // 2
    half_size = size[0] // 2

    # Hitung koordinat ROI 10x10 piksel
    small_roi_x = max(center_x - half_size, 0)
    small_roi_y = max(center_y - half_size, 0)
    small_roi_image = image[small_roi_y:small_roi_y + size[0], small_roi_x:small_roi_x + size[1]]

    return small_roi_image, (small_roi_x, small_roi_y, size[0], size[1])

# Fungsi untuk menampilkan ROI menggunakan matplotlib
def visualize_roi(roi_image, title="ROI Visualization"):
    plt.figure(figsize=(5, 5))
    plt.imshow(roi_image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

# Fungsi untuk mendeteksi dan mengenali wajah dalam gambar uji
def recognize_faces_in_image(test_image_path, known_face_encodings, known_face_names):
    test_image = cv2.imread(test_image_path)
    rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

    # Konversi gambar ke grayscale
    gray_image = convert_to_grayscale(test_image)

    # Simpan gambar grayscale
    result_filename = os.path.basename(test_image_path)
    gray_result_path = os.path.join(RESULTS_DIR, f"gray_{result_filename}")
    cv2.imwrite(gray_result_path, gray_image)
    print(f"Gambar grayscale disimpan di: {gray_result_path}")

    # Deteksi wajah
    faces = detector(rgb_image)
    for i, face in enumerate(faces):
        shape = sp(rgb_image, face)
        encoding = np.array(facerec.compute_face_descriptor(rgb_image, shape))

        # Bandingkan encoding wajah
        distances = np.linalg.norm(known_face_encodings - encoding, axis=1)
        best_match_index = np.argmin(distances)
        distance = distances[best_match_index]

        # Hitung skor persentase kecocokan (semakin kecil jarak, semakin tinggi skor)
        similarity_score = max(0, (1 - distance)) * 100  # Skor antara 0% hingga 100%
        name = "Unknown"

        if distance < 0.6:  # Threshold
            name = known_face_names[best_match_index]

        # Gambar kotak hijau untuk wajah yang terdeteksi
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())

        # Gambar 1: Kotak hijau untuk wajah yang terdeteksi
        green_box_image = test_image.copy()
        cv2.rectangle(green_box_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(green_box_image, f"{name} ({similarity_score:.2f}%)", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        green_box_path = os.path.join(RESULTS_DIR, f"green_box_{result_filename}")
        cv2.imwrite(green_box_path, green_box_image)
        print(f"Gambar dengan kotak hijau disimpan di: {green_box_path}")

        # Ekstrak ROI kecil (10x10 piksel) dari pusat wajah
        small_roi, small_roi_coords = extract_small_roi(gray_image, (x, y, w, h), size=(10, 10))
        small_roi_x, small_roi_y, small_roi_w, small_roi_h = small_roi_coords

        # Gambar 2: Kotak merah untuk ROI kecil
        red_box_image = test_image.copy()
        cv2.rectangle(red_box_image, (small_roi_x, small_roi_y), 
                      (small_roi_x + small_roi_w, small_roi_y + small_roi_h), (0, 0, 255), 2)
        red_box_path = os.path.join(RESULTS_DIR, f"red_box_{result_filename}")
        cv2.imwrite(red_box_path, red_box_image)
        print(f"Gambar dengan kotak merah disimpan di: {red_box_path}")

        # Gambar 3: Gabungan kotak hijau dan merah
        combined_image = test_image.copy()
        cv2.rectangle(combined_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(combined_image, f"{name} ({similarity_score:.2f}%)", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(combined_image, (small_roi_x, small_roi_y), 
                      (small_roi_x + small_roi_w, small_roi_y + small_roi_h), (0, 0, 255), 2)
        combined_path = os.path.join(RESULTS_DIR, f"combined_{result_filename}")
        cv2.imwrite(combined_path, combined_image)
        print(f"Gambar gabungan kotak hijau dan merah disimpan di: {combined_path}")

        # Simpan ROI kecil sebagai gambar terpisah
        small_roi_output_path = os.path.join(RESULTS_DIR, f"small_roi_{name}_{i+1}.jpg")
        cv2.imwrite(small_roi_output_path, small_roi)
        print(f"ROI kecil (10x10) disimpan di: {small_roi_output_path}")

        # Visualisasi ROI kecil menggunakan matplotlib
        visualize_roi(small_roi, title=f"ROI Kecil (10x10) untuk Wajah '{name}'")

        # Tampilkan semua elemen matriks piksel ROI kecil di konsol
        np.set_printoptions(threshold=np.inf)  # Menampilkan semua elemen array tanpa pemotongan
        print(f"Matriks piksel lengkap untuk ROI kecil (10x10) wajah '{name}':")
        print(small_roi)

# Main function
def main():
    # Muat wajah yang dikenali
    known_face_encodings, known_face_names = load_known_faces()

    # Iterasi melalui setiap gambar uji
    for filename in os.listdir(TEST_IMAGES_DIR):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            test_image_path = os.path.join(TEST_IMAGES_DIR, filename)
            print(f"Memproses gambar: {filename}")
            recognize_faces_in_image(test_image_path, known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()