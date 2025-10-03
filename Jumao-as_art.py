import cv2 as cv

screenHeight = 1080
screenWidth = 1920

canvas = cv.UMat(screenHeight, screenWidth, cv.CV_8UC3) 

cv.rectangle(canvas, (0, 0), (screenWidth, screenHeight), (200, 230, 255), -1)

center = (screenWidth // 2, screenHeight // 2)
rectangleWidth = 450
rectangleheight = 200

topLeft = (center[0] - rectangleWidth // 2, center[1] - rectangleheight // 2)
bottomRight = (center[0] + rectangleWidth // 2, center[1] + rectangleheight // 2)

cv.rectangle(canvas, topLeft, bottomRight, (180, 100, 255), -1)

text = "be |x|."
font = cv.FONT_HERSHEY_SIMPLEX
fontScale = 4
thickness = 8
color = (255, 255, 255)

(textWidth, textHeight), baseline = cv.getTextSize(text, font, fontScale, thickness)
x = center[0] - textWidth // 2
y = center[1] + textHeight // 2

cv.putText(canvas, text, (x, y), font, fontScale, color, thickness, cv.LINE_AA)

cv.namedWindow("okay", cv.WND_PROP_FULLSCREEN)
cv.setWindowProperty("okay", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
cv.imshow("okay", canvas)
# cv.imwrite('okay.png', canvas)
cv.waitKey(0)
cv.destroyAllWindows()
