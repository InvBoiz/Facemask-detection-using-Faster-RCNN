# Facemask-detection-using-Faster-RCNN

Giới thiệu bài toán

Để tránh mắc phải những sai lầm đáng tiếc như cơn đại dịch vừa rồi, mọi nơi hiện tại giờ đã coi trọng hơn việc mang khẩu trang khi ra ngoài đường. Một phần là để bảo vệ mặt mình khỏi khói bụi do môi trường xung quanh giờ có quá nhiều khí độc và khói bụi, cũng đồng thời bảo vệ chính mình và mọi người xung quanh tránh khỏi những mầm bệnh tềm tang. Từ nhu cầu đó, bài toán Nhận diện đeo khẩu trang (Facemask detection) trong các hệ thống giám sát thông minh nhầm mục đích phát hiện những người đeo hoặc không đeo khẩu trang càng được quan tâm sâu sắc hơn

Mô tả dataset

Dataset gồm 2 folder: Một folder ‘images’ chứa 853 tấm ảnh đủ loại về người đeo khẩu trang, từ những con đường đi chụp public đến những tấm ảnh chụp riêng. Một ảnh có thể có một hoặc nhiều người, có thể có hoặc không đeo khẩu trang hoặc là đeo khẩu trang không đúng cách, size ảnh không cố định, khung màu RGB. 
Một folder ‘annotations’ với số lượng file trùng với số lượng ảnh trong ‘images’, mỗi ảnh trong ‘images’ sẽ tương ứng với một file XML theo chuẩn PASCAL VOC có tên file trùng với ảnh. Bên trong mỗi file XML chứa nhiều thông tin, quan trọng nhất là một hoặc nhiều object, mỗi object tượng trưng cho một người được xác định trong tấm ảnh bao gồm:
-	Name: Nhãn của người đó (“with_mask”, “without_mask”, “mask_weared_incorrect”)
-	Bndbox: Tọa độ của bounding box (xmin, ymin, xmax, ymax)

Kiến trúc mô hình

Sử dụng mô hình Faster RCNN. Với bản Fast-RCNN, sau khi thu được các features map, rồi dùng selective search lên các features map vừa thu được  thay vì là ảnh gốc. Với Faster-RCNN, thay vì dùng selective search, mô hình được thiết kế thêm một mạng con gọi là Region Proposal Network để trích rút các vùng có khả năng chưa đối tượng của ảnh
 
Pipeline:
 
-	Đưa ảnh đầu vào qua mạng CNN để trích xuất các features map
-	Dùng RPN lên các feature maps để thu các object proposals 
-	Dùng ROI pooling lên các proposals
-	Pass các proposals này qua các lớp fully connected layer để phân loại và dự đoán bounding box cho các objects
Để trích xuất các features map thì có nhiều model khác nhau, tùy vào bài toán mà chọn các backbone có sẵn như VGG16, resnet50,… RPN hiện tại là 1 mạng fully convolution network nên không cần kích thước đầu vào phải cố định hay ảnh đầu vào của Faster-RCNN phải cố định. Và vì đầu vào kích thước ảnh không cố định nên kích thước đầu ra của RPN cũng không cố định. 
Sau khi thực hiện RoI pooling, ta thu được output features map với kích thước cố định, các feature này sẽ được flatten và đưa qua các nhánh fully connected layer để:
-	Object classification với 4 class (3 label + 1 background)
-	Bounding box regression để tune các tọa độ thu được từ ROI với ground truth bounding box
 
Quy trình huấn luyện

anno_parse() phân tích các file .xml bằng xml.etree.ElementTree, trích xuất được
-	tên nhãn (with_mask, without_mask, mask_weared_incorrect)
-	bounding box (xmin, ymin, xmax, ymax).
Lớp processDataset:
-	Read image bằng cv2 rồi chuyển sang RGB
-	Chuyển ảnh thành tensor
-	Ánh xạ label thành ID với:
1.	with_mask
2.	without_mask
3.	mask_weared_incorrect
Sau đó chia dataset theo tỷ lệ train/test là 7/3 
Sử dụng model sẵn có của Torch
Sau đó thay head bằng FastRCNNPredictor(in_features, num_classes=4) với số lượng num_classes ứng với 3 label + 1 background
Sử dụng Adam optimizer với lr=0.0005 rồi train trong 10 epoch
Đặt checkpoint và lưu checkpoint sau mỗi epoch nếu loss hiện tại là nhỏ nhất

Truong Le Minh Hieu
