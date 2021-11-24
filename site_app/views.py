from django.shortcuts import render
from .models import Target, Slot, Advertise
import json
from django.db import connection

global t_from
leng = 20
cap_time = "Not yet"
sex = 0
age_from = 0
age_to = 0

def index(request):
    global age_from, age_to, sex, leng, t_from, cap_time, cam_route

    #DB의 테이블 3개 변수에 저장
    targets = Target.objects.all()
    slots = Slot.objects.all()
    advertises = Advertise.objects.all()

    #시간대별 인구수 js에서 사용하기 위해 json형식으로 배열저장
    li = []
    for slot in slots:
        li.append(slot.cnt)
    j_li = json.dumps(li)
    #그룹별 인구수 js에서 사용하기 위해 json형식으로 배열저장
    li2 = []
    for tar in targets:
        if tar.sex != 'A':
            li2.append(tar.total_cnt)
    j_li2 = json.dumps(li2)

    link = ()
    try:    #연령대-성별에 맞는 광고중 송출횟수가 가장 적은 광고의 링크와 길이받고, 송출횟수를 올려줍니다
        cursor = connection.cursor()    #DB 접근

        strSql = """SELECT link, length
                  FROM advertise
                  WHERE tg_num IN (
                        SELECT target_num
                        FROM target
                        WHERE (sex= (%s) or sex= 'A') and age_from= (%s)
                  )ORDER BY cnt ASC LIMIT 1"""
        strSql2 = "UPDATE advertise SET cnt = cnt + 1 WHERE link = (%s)"

        cursor.execute(strSql, (sex, age_from,))  #쿼리문 strSql 실행 - 광고 링크와 길이받음
        datas = cursor.fetchall()
        link = datas[0][0]          #광고 링크
        leng = datas[0][1]          #광고 길이
        print("advertise: ", link)

        cursor.execute(strSql2, (link,))  #쿼리문 strSql2 실행 - 송출된 광고의 송출횟수 증가

        connection.commit()
        connection.close()

    except:
        connection.rollback()
        print("No face Detected, Checking next frame | Failed selecting in DB")
        cam_route = '/static/assets/img/nocam.jpg'  #나이 성별을 분석안되면 please wait 사진

    #분석된 연령대-성별 그룹 웹사이트에 출력합니다.
    if sex == 'F':
        tar_print = "{}세~{}세 여자".format(age_from, age_to)
    elif sex == 'M':
        tar_print = "{}세~{}세 남자".format(age_from, age_to)
    else:
        tar_print = "No face Detected"

    #index.html로 변수들을 넘겨주며 rendering합니다.
    return render(request, 'site_app/index.html', {"target": targets, "advers": advertises,"cam_route": cam_route,\
                                                   "time_cnt": j_li, "total_cnt": j_li2, "route": link, \
                                                   "aver_time": leng, "capture_time": cap_time, "target_print": tar_print})


def chart(request):

    #DB의 테이블 3개 변수에 저장
    targets = Target.objects.all()
    slots = Slot.objects.all()
    advertises = Advertise.objects.all()
    #시간대별 인구수 js에서 사용하기 위해 json형식으로 배열저장
    li = []
    for slot in slots:
        li.append(slot.cnt)
    j_li = json.dumps(li)
    #그룹별 인구수 js에서 사용하기 위해 json형식으로 배열저장
    li2 = []
    for tar in targets:
        if tar.sex != 'A':
            li2.append(tar.total_cnt)
    j_li2 = json.dumps(li2)

    try:
        cursor = connection.cursor()

        strSql = """SELECT link, length
                  FROM advertise
                  WHERE tg_num IN (
                        SELECT target_num
                        FROM target
                        WHERE (sex= (%s) or sex= 'A') and age_from= (%s)
                  )ORDER BY cnt ASC LIMIT 1"""
        strSql2 = "UPDATE advertise SET cnt = cnt + 1 WHERE link = (%s)"

        cursor.execute(strSql, (sex, age_from,))  # 조건에 맞는 광고 링크 받음
        datas = cursor.fetchall()  #광고 링크받음
        param = datas[0][0]

        cursor.execute(strSql2, (param,))  #송출된 광고의 송출횟수 증가

        # 송출횟수 상위 5개의 광고종류와 송출횟수
        strSql3 = """SELECT title, sum(cnt)                     
                            FROM advertise
                            GROUP BY title
                            ORDER BY sum(cnt) desc LIMIT 5 """
        cursor.execute(strSql3)
        datas1 = cursor.fetchall()

        connection.commit()
        connection.close()

        #송출횟수 상위 5개의 광고종류와 송출횟수 json형태로 저장
        kinds = [datas1[0][0], datas1[1][0], datas1[2][0], datas1[3][0], datas1[4][0]]
        kinds_cnt = [int(datas1[0][1]), int(datas1[1][1]), int(datas1[2][1]), int(datas1[3][1]), int(datas1[4][1])]
        j_li3 = json.dumps(kinds)
        j_li4 = json.dumps(kinds_cnt)

    except:
        connection.rollback()
        print("Failed selecting in DB")


    return render(request, 'site_app/charts.html', {"slots": slots, "advers": advertises, "time_cnt": j_li,\
                                                    "total_cnt": j_li2, "kinds": j_li3, "kinds_cnt": j_li4})


def table(request):

    #DB의 테이블 2개 변수에 저장
    targets = Target.objects.all()
    advertises = Advertise.objects.all()

    try:
        cursor = connection.cursor()

        strSql = """SELECT link, length
                  FROM advertise
                  WHERE tg_num IN (
                        SELECT target_num
                        FROM target
                        WHERE (sex= (%s) or sex= 'A') and age_from= (%s)
                  )ORDER BY cnt ASC LIMIT 1"""
        strSql2 = "UPDATE advertise SET cnt = cnt + 1 WHERE link = (%s)"

        cursor.execute(strSql, (sex, age_from,))  # 조건에 맞는 광고 링크 받음
        datas = cursor.fetchall()  # 광고 링크받음
        param = datas[0][0]

        cursor.execute(strSql2, (param,))  # 송출된 광고의 송출횟수 증가

        connection.commit()
        connection.close()

    except:
        connection.rollback()
        print("Failed selecting in DB")


    return render(request, 'site_app/tables.html', {"target": targets, "advers": advertises})





