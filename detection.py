import cv2
import numpy as np
import math


def order_points(points):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)

    rect[0] = pts[np.argmin(sums)]
    rect[2] = pts[np.argmax(sums)]
    rect[1] = pts[np.argmin(diffs)]
    rect[3] = pts[np.argmax(diffs)]
    return rect


def output_size(points):
    tl, tr, br, bl = points
    width_top = math.dist(tl, tr)
    width_bottom = math.dist(bl, br)
    height_left = math.dist(tl, bl)
    height_right = math.dist(tr, br)
    return int(max(width_top, width_bottom)), int(max(height_left, height_right))


def find_best_quad(edge, image_area):
    contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:80]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.01:
            continue

        perimeter = cv2.arcLength(contour, True)
        for eps in (0.01, 0.02, 0.03, 0.04, 0.06):
            corners = cv2.approxPolyDP(contour, eps * perimeter, True)
            if len(corners) == 4 and cv2.isContourConvex(corners):
                return order_points(corners.reshape(4, 2))

    return None


def scan(corDet):
    if corDet is None:
        raise ValueError("Image tidak terbaca. Periksa path gambar di cv2.imread().")

    org = corDet.copy()
    org1 = corDet.copy()
    height, width, channel = corDet.shape
    grey  = cv2.cvtColor(corDet, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    edge = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel, iterations=1)

    cont, cod = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    Dcont = cv2.drawContours(org1, cont, -1, (0,0,255),3)
    corLis = find_best_quad(edge, height * width)

    if corLis is None:
        msg = "Tidak ditemukan 4 titik. Coba foto dengan tepi lebih jelas atau pilih manual."
        cv2.putText(corDet, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        return [org, grey, edge, Dcont, corDet, org.copy(), org.copy()]

    Owidth, Oheight = output_size(corLis)
    Owidth = max(1, Owidth)
    Oheight = max(1, Oheight)

    for index, point in enumerate(corLis.astype(int)):
        x, y = point
        xy = f"P{index + 1} ({x},{y})"
        cv2.putText(corDet, xy, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 2)
        corDet = cv2.circle(corDet, (x,y), 5, (0,0,255),cv2.FILLED)
    cv2.polylines(corDet, [corLis.astype(int)], True, (0,255,0), 3)

    pts1 = np.float32(corLis)
    pts2 = np.float32([[0,0],[Owidth - 1,0],[Owidth - 1,Oheight - 1],[0,Oheight - 1]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    wap = cv2.warpPerspective(org, matrix, (Owidth,Oheight))
    out = wap.copy()
    return [org, grey, edge, Dcont, corDet, wap, out]

def display(Pimg,p=1,s=1):
	org = Pimg[0]
	grey = cv2.cvtColor(Pimg[1],cv2.COLOR_GRAY2BGR)
	edge = cv2.cvtColor(Pimg[2],cv2.COLOR_GRAY2BGR)
	Dcont = Pimg[3]
	corDet = Pimg[4]
	wap = Pimg[5]
	height, width, channel = org.shape
	wap_display = cv2.resize(wap, (width, height))
	Oheight, Owidth, Ochanel = Pimg[6].shape
	meg1 = np.concatenate((org,grey,edge),1)
	meg2 = np.concatenate((Dcont,corDet,wap_display),1)
	meg3 = np.concatenate((meg1,meg2),0)
	Mheight, Mwidth, Mchannel = meg3.shape
	ip = cv2.resize(meg3, (int(Mwidth*p),int(Mheight*p)))
	io = cv2.resize(Pimg[6], (int(Owidth*s),int(Oheight*s)))
	return ip, io

if __name__=='__main__':
	img = cv2.imread("D:/Uni/Academic/Sem 6/Pengolahan Analisis Citra & Digital/UAS/receipt.png") # enter your image path here
	ip, io = display(scan(img),0.5,1)
	cv2.imshow("process",ip)
	cv2.imshow("output", io)
	cv2.waitKey(0)
	# cv2.imwrite("output1.jpg",io
