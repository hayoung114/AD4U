# 인공지능 모델 사용하는 main_OpenCV_SSD.py
import threading                #thread를 사용하기 위한 라이브러리
import cv2 as cv                #OpenCV를 사용하기 위한 라이브러리
import time                     #시간을 재기 위한 라이브러리
import argparse                 #명령행 파싱 모듈
from collections import Counter #그룹별 인구수 구하기 위한 라이브러리
from site_app import views      #views.py의 변수들 값 접근
from django.db import connection    #DB접근
from tensorflow.keras.models import load_model  #학습시킨 모델 load
import numpy as np


#성별 인식 모델입니다.
model = load_model(
    'C:/Users/USER/PycharmProjects/Project_Advertise/ai_project/ai_models/final_Continue_our_large_model_epoch_5.h5',
    compile=True)


#웹캠 영상을 띄우는 함수
def Webcam():
    global frame
    capture = cv.VideoCapture(0)  #웹캠을 객체로 만듭니다.
    capture.set(3, 640)  #픽셀길이 가로 640
    capture.set(4, 480)  #픽셀길이 세로 480

    while True:  #'q'키를 누를 때까지 반복
        ret, frame = capture.read()  #카메라로부터 영상 하나 읽어옵니다.
        cv.imshow('frame', frame)  # 영상을 window 에 표시합니다.
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


#thread로 실행될 project 함수
def Project():
    global frame

    capture_counter = 1
    start_time = time.time()

    while True:  #'q'키를 누를 때까지 반복
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if time.time() - start_time >= views.leng:  #<---- (광고시간1)초 뒤에 캡쳐합니다.
            start_time = time.time()
            img_name = "C:/Users/USER/PycharmProjects/Project_Advertise/ai_project/img/opencv_frame.jpg"
            cv.imwrite(img_name, frame)  #영상에서 캡쳐한 이미지를 저장합니다.

            print("_______________{} written!________________".format(capture_counter))

            #캡쳐한 시간을 현재 시간으로 html에 넘겨주기 위한 코드
            capture_time = time.localtime()
            views.cap_time = "Updated {0:04d}/{1:02d}/{2:02d} at {3:02d}:{4:02d}".format(capture_time.tm_year,\
                            capture_time.tm_mon, capture_time.tm_mday, capture_time.tm_hour, capture_time.tm_min)

            #현재시각에 맞는 시간대 테이블 row접근
            s_num = capture_time.tm_hour//2 + 1     # 시간대 번호 slot_num

            #2시간마다 시간대별 인구분석을 위해 인구 수 0으로 초기화하고 그전 시간대의 최빈 그룹 저장
            if capture_time.tm_hour % 2 == 0 and capture_time.tm_min == 0:
                if s_num != 1:
                    ss_num = s_num-1
                else:
                    ss_num = 12
                try:
                    cursor = connection.cursor()     #DB 접근
                    strSql6 = """select target_num from target 
                                where slot_cnt IN (select max(slot_cnt) from target) 
                                order by slot_cnt ASC LIMIT 1"""
                    strSql7 = "update slot set tg_num = (%s) where slot_num = (%s)"
                    strSql8 = "UPDATE target SET slot_cnt = 0 WHERE target_num > 0"
                    cursor.execute(strSql6)  # 최대값 받음
                    datas = cursor.fetchall()
                    param = datas[0][0]
                    print(param)
                    cursor.execute(strSql7, (param, ss_num,))
                    cursor.execute(strSql8)

                    connection.commit()
                    connection.close()

                except:
                    connection.rollback()
                    print("Failed selecting in DB6")

            #프로그램이 받고 싶은 명령행 옵션을 지정하기 위해 사용합니다.
            parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
            parser.add_argument('--input',
                                help='Path to input image or video file. Skip this argument to capture frames from a camera.')
            parser.add_argument("--device", default="cpu", help="Device to inference on")

            args = parser.parse_args(['--input', img_name])

            #SSD를 이용한 얼굴인식 모델명을 인자로 넘겨주기 위해 변수들에 저장합니다.
            faceProto = "C:/Users/USER/PycharmProjects/Project_Advertise/ai_project/ai_models/opencv_face_detector.pbtxt"
            faceModel = "C:/Users/USER/PycharmProjects/Project_Advertise/ai_project/ai_models/opencv_face_detector_uint8.pb"

            #연령대 인식 모델입니다.
            ageProto = "C:/Users/USER/PycharmProjects/Project_Advertise/ai_project/ai_models/age_deploy.prototxt"
            ageModel = "C:/Users/USER/PycharmProjects/Project_Advertise/ai_project/ai_models/age_net.caffemodel"

            #bolb 에서 빼야 할 RGB 값들을 MODEL_MEAN_VALUES에 저장합니다.
            MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

            #저희가 분류 할 연령대와 성별입니다.
            ageList = ['(1, 7)', '(20, 25)', '(14, 19)', '(20, 25)', '(26, 33)', '(26, 33)', '(51, 70)', '(70, 100)']
            genderList = ['M', 'F']

            #얼굴인식 모델과 사진을 받아와서 얼굴부분을 인식하고 박스를 그리는 함수 선언 부분 입니다.
            def getFaceBox(net, frame, conf_threshold=0.7):         #정확도가 0.7보다 높으면 얼굴로 인식합니다.
                frameOpencvDnn = frame.copy()                       #원본 사진의 정보를 가져와 가로, 세로 크기를 가져옵니다.
                frameHeight = frameOpencvDnn.shape[0]
                frameWidth = frameOpencvDnn.shape[1]
                blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

                net.setInput(blob)              #네트워크 입력을 설정합니다.
                detections = net.forward()      #네트워크 추론을 합니다.
                bboxes = []                     #얼굴 좌표값들을 저장할 리스트

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]

                    if confidence > conf_threshold:                 #얼굴이라고 70% 이상으로 예상되면 좌표를 반환합니다.
                        x1 = int(detections[0, 0, i, 3] * frameWidth)
                        y1 = int(detections[0, 0, i, 4] * frameHeight)
                        x2 = int(detections[0, 0, i, 5] * frameWidth)
                        y2 = int(detections[0, 0, i, 6] * frameHeight)
                        bboxes.append([x1, y1, x2, y2])
                        cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8) #얼굴 박스를 그립니다.
                return frameOpencvDnn, bboxes


            # 얼굴, 연령대 모델을 각각 불러옵니다.
            ageNet = cv.dnn.readNet(ageModel, ageProto)
            faceNet = cv.dnn.readNet(faceModel, faceProto)

            # cpu device를 이용할 경우입니다,
            if args.device == "cpu":
                ageNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)

                faceNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)

                print("Using CPU device")

            #gpu device를 이용할 경우입니다.
            elif args.device == "gpu":
                ageNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
                ageNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

                faceNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
                faceNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

                print("Using GPU device")


            #위에서 선언한 getFaceBox 함수로 얼굴의 좌표를 bboxes에 받아옵니다.
            frameFace, bboxes = getFaceBox(faceNet, frame)


            if not bboxes:
                print("No face Detected, Checking next frame")

                try:    #얼굴이 인식되지 않을 경우 현재 시간대 최다그룹의 광고를 defult광고로 송출합니다.
                    cursor = connection.cursor()    #DB 접근
                    strSql9 = "select tg_num from slot where slot_num = (%s)"
                    strSql10 = "select age_from, age_to, sex from target where target_num = (%s)"

                    cursor.execute(strSql9, (s_num,))
                    datas1 = cursor.fetchall()
                    param1 = datas1[0][0]

                    cursor.execute(strSql10, (param1,))
                    datas2 = cursor.fetchall()
                    age_f = datas2[0][0]
                    age_t = datas2[0][1]
                    gender = datas2[0][2]

                    connection.commit()
                    connection.close()

                    views.age_from = age_f
                    views.age_to = age_t
                    views.sex = gender

                except:
                    connection.rollback()
                    print("Failed selecting in DB5")


                continue

            #연령대, 성별의 최빈값을 구하기 위한 딕셔너리와 리스트 입니다.
            dic={}
            result=[]

            t = time.time()
            padding = 20

            #얼굴이 인식됐을 경우
            for bbox in bboxes:
                #얼굴 구역을 잘라냅니다.
                face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                       max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

                #얼굴의 면적을 구합니다. 추후에 가까이 있는 사람을 고르기 위함입니다.
                area = abs(bbox[0] - bbox[2]) * abs(bbox[1] - bbox[3])

                #얼굴부분만 인식시켜서 연령대를 파악하기 위해 네트워크를 설정합니다.
                blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                #성별을 학습하고 결과를 출력합니다.
                face = cv.resize(face, dsize=(224, 224))
                face = np.reshape(face, [1, 224, 224, 3])
                prediction = model.predict(face)
                print(prediction)
                gender_predict = float(prediction)

                if gender_predict > 0.5:
                    gender = 'M'
                    conf = (gender_predict-0.5) * 2
                else:
                    gender = 'F'
                    conf = (0.5-gender_predict) * 2
                print("Gender : {}, conf = {:.3f}".format(gender, conf))

                #연령대를 분석하고 결과를 출력합니다.
                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

                tt = time.time() - t    #분석 시간 구하기 위해 시간저장

                #연령대와 성별을 리스트에 저장합니다.
                result.append((age, gender))

                #최빈값을 구할 때 동점이 나올 경우 얼굴 크기로 가까이 있는 사람을 기준으로 잡기 위한 얼굴면적 저장입니다.
                if (age, gender) not in dic.keys():
                    dic[(age, gender)] = [area]
                else:
                    dic.get((age, gender)).append(area)

                #사진에 분석된 라벨을 넣습니다.
                label = "{}, {}".format(gender, age)
                cv.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                           cv.LINE_AA)


                if age[2] == ',':
                    age_fr = int(age[1])
                else:
                    age_fr = int(age[1:3])
                try:    #인식된 얼굴마다 연령대와 성별 그릅 수를 증가시킵니다.
                    cursor = connection.cursor()    #DB 접근
                    strSql4 = "UPDATE target SET total_cnt = total_cnt + 1 WHERE sex= (%s) and age_from= (%s)"
                    strSql5 = "UPDATE target SET slot_cnt = slot_cnt + 1 WHERE sex= (%s) and age_from= (%s)"
                    cursor.execute(strSql4, (gender, age_fr,))
                    cursor.execute(strSql5, (gender, age_fr,))
                    connection.commit()
                    connection.close()

                except:
                    connection.rollback()
                    print("Failed selecting in DB4")

            lst = list(dict(Counter(result)).values())     #인식된 그룹별 인구수 리스트
            dic2 = dict(Counter(result))                #인식된 사람들 {그룹:수} 딕셔너리
            f_cnt = len(bboxes)                         #인식된 사람 수

            try:    #인식된 사람 수 만큼 시간대 테이블의 인구수를 증가 시킵니다.
                cursor = connection.cursor()    #DB 접근
                strSql3 = "UPDATE slot SET cnt = cnt + (%s) WHERE slot_num = (%s)"

                cursor.execute(strSql3, (f_cnt, s_num,))
                connection.commit()
                connection.close()

            except:
                connection.rollback()
                print("Failed selecting in DB3")

            #최빈그룹 중 가장 가까이 있는 사람을 구합니다.
            max_area = 0
            ans = 0
            for k, v in dic2.items():
                if v == max(lst):
                    if max_area < max(dic.get(k)):
                        max_area = max(dic.get(k))
                        ans = k

            age, gender = ans

            #DB의 target table에 접근하기 위해 views.py의 age_from, age_to 변수에 대입
            if age[2] == ',':
                age_f = int(age[1])
                age_t = int(age[3:-1])
            else:
                age_f = int(age[1:3])
                age_t = int(age[4:-1])
            views.age_from = age_f
            views.age_to = age_t
            views.sex = gender

            #분석된 사진을 저장합니다.
            frame_name = "C:/Users/USER/PycharmProjects/Project_Advertise/site_app/static/assets/img/cam.jpg"
            cv.imwrite(frame_name, frameFace)  #이미지를 저장합니다.
            cv.imshow("Age Gender Demo", frameFace) #연령대-성별이 써있는 캡쳐사진을 띄웁니다.
            views.cam_route = '/static/assets/img/cam.jpg'

            #실행되는 시간을 출력합니다.
            print("model time : {:.3f}".format(tt))    #연령대,성별 분석시간
            print("total time : {:.3f}".format(time.time() - start_time))  #총 시간

            capture_counter += 1
