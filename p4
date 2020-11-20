# project  4 : 얼굴에 썬글라스 씌우기

img_copy = img.copy()
h, w = img_copy.shape[:2]

# sunglasses이미지로 마스크체크
sg_image = cv2.imread('sunglasses.png') # sg_image = sunglasses이미지
image_gray = cv2.cvtColor(sg_image, cv2.COLOR_BGR2GRAY)

_,e_mask = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY_INV) # 선글라스 이미지 마스크
sg_mask = cv2.merge((e_mask, e_mask, e_mask))
sg_masked_image = cv2.bitwise_and(sg_image, sg_mask)       
sgh, sgw = sg_masked_image.shape[:2]


ree_x = np.int(sgw*0.15);  lee_x = np.int(sgw*(1-0.13))  # 왼쪽, 오른쪽 눈 위치 대응점 좌표
ee_y = np.int(sgh*0.45)  # y를 눈 지점으로  설정
eyecornerSrc = [(ree_x, ee_y), (lee_x, ee_y)]

cv2.circle(sg_masked_image, eyecornerSrc[0], int(sgh*0.03), (0,0,255), -1)
cv2.circle(sg_masked_image, eyecornerSrc[1], int(sgh*0.03), (0,0,255), -1)

plt.figure(figsize=(10,10))
plt.subplot(131); plt.imshow(sg_image); plt.title('sunglassses')
plt.subplot(132); plt.imshow(sg_mask); plt.title('sg_mask')
plt.subplot(133); plt.imshow(sg_masked_image[:,:,::-1]); plt.title('sg_masked_image')

# im_copy와 같은 크기 zero 영상 만듦.   안경 마스크, 영상을 중첩시킬 예정
sg_maskDst = np.zeros_like(img_copy)
sg_maskedImageDst = np.zeros_like(img_copy)

for i in range(0, len(faceRects)):
  
    landmarks = landmarkDetector(img_copy, faceRects[i])
    point2 = (landmarks.part(36).x, landmarks.part(36).y) # 오른쪽 눈 랜드마크 36
    point0 = (landmarks.part(45).x, landmarks.part(45).y) # 왼쪽 눈 랜드마크 45
    eyecornerDst = [point2, point0]
  
   #두 대응점으로부터 썬글라스가 원 영상으로 이동할 변한행렬 계산
    ret = cv2.estimateAffinePartial2D(np.array([eyecornerSrc]), np.array([eyecornerDst]))
    xform = ret[0]
    print('xform :', xform)

  # 변환 적용
    xformed_mask = cv2.warpAffine(s_mask, xform, (w, h))
    xformed_maskedImage = cv2.warpAffine(sg_masked_image, xform, (w, h))

  
    im_vis = im.copy()
    x1 = faceRects[i].left()
    y1 = faceRects[i].top()
    x2 = faceRects[i].right()
    y2 = faceRects[i].bottom()
    cv2.rectangle(im_vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.circle(im_vis, point2, 3, (0,0,255), -1)
    cv2.circle(im_vis, point0, 3, (0,0,255), -1)


    plt.figure(figsize=(10,10))
    plt.subplot(131); plt.imshow(im_vis[:,:,::-1]); plt.title('face'+np.str(i))
    plt.subplot(132); plt.imshow(xformed_mask[:,:,::-1]); plt.title('mask'+np.str(i))
    plt.subplot(133); plt.imshow(xformed_maskedImage[:,:,::-1]); plt.title('maskedImage'+np.str(i))
    
    sg_maskDst = sg_maskDst + xformed_mask 
    sg_maskedImageDst = sg_maskedImageDst + xformed_maskedImage
    
plt.figure(figsize=(10,10))
plt.subplot(121); plt.imshow(sg_maskDst[:,:,::-1]); plt.title('sg_maskDst')
plt.subplot(122); plt.imshow(sg_maskedImageDst[:,:,::-1]); plt.title('sg_maskedImageDst')

dst_blend = (img_copy*0.7+ sg_maskedImageDst*0.3).astype(np.uint8) # alpha-blending. lena+rocket
sg_mask_float = sg_maskDst / 255
sg_mask_Invert = 1-sg_maskDst
dst_obj = (dst_blend * sg_mask_float).astype(np.uint8)
dst_back = (img_copy * (1-sg_mask_float)).astype(np.uint8)
dst_total = dst_obj + dst_back

plt.figure(figsize=(10,10))
plt.subplot(311); plt.imshow(im[:,:,::-1]); plt.title('input image')
plt.subplot(312); plt.imshow(s_image[:,:,::-1]); plt.title('sunglasses')
plt.subplot(313); plt.imshow(dst_total[:,:,::-1]);  plt.title('result')
