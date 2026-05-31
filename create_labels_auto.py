# improved_label_generation.py
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import time
import shutil

class ImprovedLabelGenerator:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.setup_directories()
        
    def setup_directories(self):
        """Create all necessary directories"""
        directories = [
            "data/raw",
            "outputs/noisy_images",
            "outputs/denoised_images",
            "data/yolo_dataset/train/images",
            "data/yolo_dataset/train/labels",
        ]
        
        for dir_path in directories:
            (self.base_path / dir_path).mkdir(parents=True, exist_ok=True)
        
        print("✓ Directories created")
    
    def create_sample_images(self, num_samples=20):
        """Create realistic sample face images using actual face dataset or better drawings"""
        sample_dir = self.base_path / "data/raw"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating {num_samples} sample face images...")
        
        # Download a sample face image if needed (using a simple approach)
        # Alternatively, create more realistic face-like shapes
        
        for i in range(num_samples):
            # Create a more realistic face-like image
            img = np.ones((300, 300, 3), dtype=np.uint8) * 245  # Light gray background
            
            # Face oval
            face_center = (150, 150)
            face_axes = (70, 90)
            cv2.ellipse(img, face_center, face_axes, 0, 0, 360, (220, 180, 140), -1)
            
            # Eyes
            left_eye = (120, 130)
            right_eye = (180, 130)
            cv2.circle(img, left_eye, 10, (0, 0, 0), -1)
            cv2.circle(img, right_eye, 10, (0, 0, 0), -1)
            cv2.circle(img, left_eye, 3, (255, 255, 255), -1)
            cv2.circle(img, right_eye, 3, (255, 255, 255), -1)
            
            # Nose
            nose_points = np.array([[145, 150], [150, 170], [155, 150]], np.int32)
            cv2.fillPoly(img, [nose_points], (180, 140, 100))
            
            # Mouth
            cv2.ellipse(img, (150, 200), (25, 15), 0, 0, 180, (100, 80, 60), -1)
            
            # Eyebrows
            cv2.line(img, (105, 115), (135, 120), (0, 0, 0), 3)
            cv2.line(img, (165, 120), (195, 115), (0, 0, 0), 3)
            
            # Add some random variation
            if i % 3 == 0:
                # Add glasses for some faces
                cv2.rectangle(img, (100, 115), (140, 145), (0, 0, 0), 2)
                cv2.rectangle(img, (160, 115), (200, 145), (0, 0, 0), 2)
                cv2.line(img, (140, 130), (160, 130), (0, 0, 0), 2)
            
            # Save image
            cv2.imwrite(str(sample_dir / f"face_sample_{i:03d}.jpg"), img)
        
        print(f"✓ Created {num_samples} sample images in {sample_dir}")
        return list(sample_dir.glob("*.jpg"))
    
    def add_noise_to_images(self, images):
        """Add noise to images"""
        print("\n📊 Adding noise to images...")
        
        noisy_dir = self.base_path / "outputs/noisy_images"
        
        def add_gaussian(img, sigma=15):
            noise = np.random.normal(0, sigma, img.shape)
            noisy = img.astype(np.float32) + noise
            return np.clip(noisy, 0, 255).astype(np.uint8)
        
        def add_salt_pepper(img, amount=0.01):
            noisy = img.copy()
            h, w = img.shape[:2]
            num_salt = int(amount * h * w)
            
            # Add salt (white)
            coords = [np.random.randint(0, i, num_salt) for i in [h, w]]
            noisy[coords[0], coords[1]] = 255
            
            # Add pepper (black)
            num_pepper = int(amount * h * w)
            coords = [np.random.randint(0, i, num_pepper) for i in [h, w]]
            noisy[coords[0], coords[1]] = 0
            return noisy
        
        noise_types = {
            'gaussian': add_gaussian,
            'salt_pepper': add_salt_pepper
        }
        
        total_images = 0
        for noise_name, noise_func in noise_types.items():
            noise_type_dir = noisy_dir / noise_name
            noise_type_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in tqdm(images, desc=f"Adding {noise_name} noise"):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                noisy_img = noise_func(img)
                output_path = noise_type_dir / f"{noise_name}_{img_path.name}"
                cv2.imwrite(str(output_path), noisy_img)
                total_images += 1
        
        print(f"✓ Created {total_images} noisy images")
        return noisy_dir
    
    def apply_denoising_filters(self, noisy_dir):
        """Apply denoising filters to noisy images"""
        print("\n📊 Applying denoising filters...")
        
        denoised_dir = self.base_path / "outputs/denoised_images"
        
        def apply_gaussian(img, ksize=3):
            return cv2.GaussianBlur(img, (ksize, ksize), 1.0)
        
        def apply_median(img, ksize=3):
            return cv2.medianBlur(img, ksize)
        
        def apply_bilateral(img):
            return cv2.bilateralFilter(img, 9, 75, 75)
        
        filters = {
            'gaussian': apply_gaussian,
            'median': apply_median,
            'bilateral': apply_bilateral
        }
        
        total_denoised = 0
        for noise_type in noisy_dir.iterdir():
            if not noise_type.is_dir():
                continue
            
            for filter_name, filter_func in filters.items():
                filter_dir = denoised_dir / noise_type.name / filter_name
                filter_dir.mkdir(parents=True, exist_ok=True)
                
                for img_path in tqdm(noise_type.glob("*.*"), 
                                    desc=f"Applying {filter_name} to {noise_type.name}"):
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    denoised = filter_func(img)
                    output_path = filter_dir / img_path.name
                    cv2.imwrite(str(output_path), denoised)
                    total_denoised += 1
        
        print(f"✓ Created {total_denoised} denoised images")
        return denoised_dir
    
    def detect_faces_with_multiple_methods(self, image_path):
        """Try multiple face detection methods"""
        img = cv2.imread(str(image_path))
        if img is None:
            return []
        
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        all_faces = []
        
        # Method 1: Haar Cascade (default)
        haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        all_faces.extend(faces)
        
        # Method 2: Alternative Haar Cascade
        alt_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        )
        faces_alt = alt_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
        )
        all_faces.extend(faces_alt)
        
        # Method 3: Profile face detector for side faces
        profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )
        faces_profile = profile_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
        )
        all_faces.extend(faces_profile)
        
        # Remove duplicates (simple approach - check overlap)
        unique_faces = []
        for face in all_faces:
            x, y, w, h = face
            is_duplicate = False
            for existing in unique_faces:
                ex, ey, ew, eh = existing
                if abs(x - ex) < 20 and abs(y - ey) < 20:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_faces.append(face)
        
        # Convert to YOLO format
        yolo_boxes = []
        for (x, y, w, h) in unique_faces:
            center_x = (x + w/2) / width
            center_y = (y + h/2) / height
            norm_w = w / width
            norm_h = h / height
            yolo_boxes.append(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
        
        return yolo_boxes
    
    def process_single_image(self, img_path, output_label_dir):
        """Process a single image for threading"""
        label_path = output_label_dir / f"{img_path.stem}.txt"
        if label_path.exists() and label_path.stat().st_size > 0:
            return 1, img_path
        
        boxes = self.detect_faces_with_multiple_methods(img_path)
        
        if boxes:
            with open(label_path, 'w') as f:
                f.write('\n'.join(boxes))
            return 1, img_path
        return 0, img_path
    
    def generate_labels(self, denoised_dir, max_workers=None):
        """Generate YOLO labels for denoised images"""
        print("\n📊 Generating YOLO labels...")
        
        yolo_dataset_path = self.base_path / "data/yolo_dataset"
        labels_dir = yolo_dataset_path / 'train' / 'labels'
        images_dir = yolo_dataset_path / 'train' / 'images'
        
        # Collect all denoised images
        all_images = []
        for noise_type in denoised_dir.iterdir():
            if not noise_type.is_dir():
                continue
            for filter_name in noise_type.iterdir():
                if not filter_name.is_dir():
                    continue
                for img_path in filter_name.glob("*.*"):
                    all_images.append(img_path)
        
        print(f"Found {len(all_images)} images to process")
        
        if len(all_images) == 0:
            print("❌ No images found!")
            return 0
        
        if max_workers is None:
            max_workers = min(mp.cpu_count(), 4)
        
        print(f"Using {max_workers} threads")
        
        # Process using ThreadPoolExecutor
        total_labeled = 0
        images_with_faces = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_single_image, img_path, labels_dir) 
                      for img_path in all_images]
            
            for future in tqdm(futures, desc="Detecting faces"):
                result, img_path = future.result()
                if result:
                    total_labeled += 1
                    images_with_faces.append(img_path)
        
        # Copy images that have faces
        for img_path in images_with_faces:
            dest_name = f"{img_path.parent.parent.name}_{img_path.parent.name}_{img_path.name}"
            dest_path = images_dir / dest_name
            shutil.copy2(img_path, dest_path)
        
        print(f"\n✓ Labeled {total_labeled} out of {len(all_images)} images")
        return total_labeled
    
    def create_data_yaml(self):
        """Create data.yaml for YOLO training"""
        import yaml
        
        yaml_content = {
            'path': str(self.base_path / 'data/yolo_dataset'),
            'train': 'train/images',
            'val': 'train/images',  # Use same for now, you can split later
            'test': 'train/images',
            'nc': 1,
            'names': ['face']
        }
        
        yaml_path = self.base_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"✓ Created data.yaml at {yaml_path}")
        return yaml_path
    
    def run_complete_pipeline(self):
        """Run the complete pipeline"""
        print("\n" + "="*60)
        print("IMPROVED FACE DETECTION PIPELINE")
        print("="*60)
        
        start_time = time.time()
        
        # Step 1: Create sample images
        images = self.create_sample_images(30)
        print(f"\n✓ Using {len(images)} images")
        
        # Step 2: Add noise
        noisy_dir = self.add_noise_to_images(images)
        
        # Step 3: Apply denoising
        denoised_dir = self.apply_denoising_filters(noisy_dir)
        
        # Step 4: Generate labels
        num_labeled = self.generate_labels(denoised_dir)
        
        # Step 5: Create data.yaml
        self.create_data_yaml()
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED!")
        print("="*60)
        print(f"✓ Total original images: {len(images)}")
        print(f"✓ Total denoised images: {num_labeled * 3} (3 filters applied)")
        print(f"✓ Images with faces detected: {num_labeled}")
        print(f"✓ Detection rate: {num_labeled/len(images)*100:.1f}%")
        print(f"✓ Total time: {elapsed:.2f} seconds")
        
        return num_labeled

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    pipeline = ImprovedLabelGenerator(r"C:\Users\DELL\Desktop\Projet_FaceDenoising")
    num_labeled = pipeline.run_complete_pipeline()
    
    if num_labeled > 0:
        print("\n✅ Success! Labels generated successfully!")
        print("\nNext steps:")
        print("1. Train YOLO model:")
        print("   python train_yolo.py")
        print("\n2. Or use the labeled dataset for face detection training")
    else:
        print("\n⚠️  Still no faces detected. Alternative solutions:")
        print("1. Download real face images from online sources")
        print("2. Use a pre-trained face detection model")
        print("3. Install MTCNN: pip install mtcnn")
        print("4. Use your own face images in 'data/raw' folder")