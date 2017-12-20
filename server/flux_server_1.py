import pymysql.cursors
import socket
import sys

MAX = 100
Massenger = ''
MSG1 = '' # (0,0) (1,0) (2,0) (3,0) (4,0)
MSG2 = '' # (4,0) (5,0) (6,0) (7,0) (8,0)
MSG3 = '' # (0,1) (0,2) (0,3) (0,4) (0,5)
MSG4 = '' # (4,0) (4,1) (4,2) (4,3) (4,4)
MSG5 = '' # (8,0) (8,1) (8,2) (8,3) (8,4)
MSG6 = '' # (0,4) (1,4) (2,4) (3,4) (4,4)
MSG7 = '' # (4,4) (5,4) (6,4) (7,4) (8,4)
INITMSG = '9:0:0:1,8:0:0:1,7:0:0:1,6:0:0:1,5:0:0:1,4:0:0:1,3:0:0:1,2:0:0:1,1:0:0:1,1:1:0:1,1:2:0:1,1:3:0:1,1:4:0:1,5:1:0:1,5:2:0:1,5:3:0:1,5:4:0:1,4:4:0:1,3:4:0:1,2:4:0:1'
INITMAP = '9:0:0:1,8:0:0:0,7:0:0:0,6:0:0:0,5:0:0:0,4:0:0:0,3:0:0:0,2:0:0:0,1:0:0:0,1:1:0:0,1:2:0:0,1:3:0:0,1:4:0:0,5:1:0:0,5:2:0:0,5:3:0:0,5:4:0:0,4:4:0:0,3:4:0:0,2:4:0:0'

preData = '9:0:0:0'
postX = '0'
postY = '0'
preX = '9'
preY = '0'
#isCarbeing = False
direction = [0,0,0,1]
carX = '0'
carY = '0'

def parseMap():
    DBCONN = pymysql.connect(host = 'localhost',
                           user = 'root',
                           password = None,
                           db = 'car_map_data',
                           charset = 'utf8')
    with DBCONN.cursor() as cursor:
        parkinglot = [[0 for col in range(6)] for row in range(10)]
        i = 0
        sql = 'SELECT y, park, car, obstacle, parked FROM map WHERE x = %s'
        for i in range(0,10):
            cursor.execute(sql, (str(i)),)
            results = cursor.fetchall()
            for row in results:
                if row[3] == 1: # obstacle
                    parkinglot[i][row[0]] = 5
                elif row[2] == 1: # car
                    parkinglot[i][row[0]] = 4
                elif row[4] == 1: # park
                    parkinglot[i][row[0]] = 3
                elif row[1] == 1:
                    parkinglot[i][row[0]] = 2
                else:
                    parkinglot[i][row[0]] = 0
        DBCONN.close()
        return parkinglot
        
parkinglot = parseMap()


x = 9 #차량 위치
y = 0 #차량 위치

destination = 0

X = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #목적지 위치
Y = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #목적지 위치

pos_num = 0  #목적지 개수
node_num = 0 #목적지까지 가는데 필요한 위치 개수

des_node = [
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0]
    ]        #목적지 좌표(x,y,최종 목적지를 나타내는 번호)

tracking_node = [
    [[0  ,MAX,MAX,MAX,MAX,MAX],
     [MAX,0  ,4  ,4  ,MAX,MAX],
     [MAX,4  ,0  ,MAX,4  ,MAX],
     [MAX,4  ,MAX,0  ,4  ,MAX],
     [MAX,MAX,4  ,4  ,0  ,MAX],
     [MAX,MAX,MAX,MAX,MAX,0  ]],

    [[0  ,MAX,MAX,MAX,MAX,MAX],
     [MAX,0  ,4  ,4  ,MAX,MAX],
     [MAX,4  ,0  ,MAX,4  ,MAX],
     [MAX,4  ,MAX,0  ,4  ,MAX],
     [MAX,MAX,4  ,4  ,0  ,MAX],
     [MAX,MAX,MAX,MAX,MAX,0  ]],

    [[0  ,MAX,MAX,MAX,MAX,MAX],
     [MAX,0  ,4  ,4  ,MAX,MAX],
     [MAX,4  ,0  ,MAX,4  ,MAX],
     [MAX,4  ,MAX,0  ,4  ,MAX],
     [MAX,MAX,4  ,4  ,0  ,MAX],
     [MAX,MAX,MAX,MAX,MAX,0  ]],

    [[0  ,MAX,MAX,MAX,MAX,MAX],
     [MAX,0  ,4  ,4  ,MAX,MAX],
     [MAX,4  ,0  ,MAX,4  ,MAX],
     [MAX,4  ,MAX,0  ,4  ,MAX],
     [MAX,MAX,4  ,4  ,0  ,MAX],
     [MAX,MAX,MAX,MAX,MAX,0  ]],

    [[0  ,MAX,MAX,MAX,MAX,MAX],
     [MAX,0  ,4  ,4  ,MAX,MAX],
     [MAX,4  ,0  ,MAX,4  ,MAX],
     [MAX,4  ,MAX,0  ,4  ,MAX],
     [MAX,MAX,4  ,4  ,0  ,MAX],
     [MAX,MAX,MAX,MAX,MAX,0  ]],

    [[0  ,MAX,MAX,MAX,MAX,MAX],
     [MAX,0  ,4  ,4  ,MAX,MAX],
     [MAX,4  ,0  ,MAX,4  ,MAX],
     [MAX,4  ,MAX,0  ,4  ,MAX],
     [MAX,MAX,4  ,4  ,0  ,MAX],
     [MAX,MAX,MAX,MAX,MAX,0  ]],

    [[0  ,MAX,MAX,MAX,MAX,MAX],
     [MAX,0  ,4  ,4  ,MAX,MAX],
     [MAX,4  ,0  ,MAX,4  ,MAX],
     [MAX,4  ,MAX,0  ,4  ,MAX],
     [MAX,MAX,4  ,4  ,0  ,MAX],
     [MAX,MAX,MAX,MAX,MAX,0  ]],

    [[0  ,MAX,MAX,MAX,MAX,MAX],
     [MAX,0  ,4  ,4  ,MAX,MAX],
     [MAX,4  ,0  ,MAX,4  ,MAX],
     [MAX,4  ,MAX,0  ,4  ,MAX],
     [MAX,MAX,4  ,4  ,0  ,MAX],
     [MAX,MAX,MAX,MAX,MAX,0  ]],

    [[0  ,MAX,MAX,MAX,MAX,MAX],
     [MAX,0  ,4  ,4  ,MAX,MAX],
     [MAX,4  ,0  ,MAX,4  ,MAX],
     [MAX,4  ,MAX,0  ,4  ,MAX],
     [MAX,MAX,4  ,4  ,0  ,MAX],
     [MAX,MAX,MAX,MAX,MAX,0  ]],

    [[0  ,MAX,MAX,MAX,MAX,MAX],
     [MAX,0  ,4  ,4  ,MAX,MAX],
     [MAX,4  ,0  ,MAX,4  ,MAX],
     [MAX,4  ,MAX,0  ,4  ,MAX],
     [MAX,MAX,4  ,4  ,0  ,MAX],
     [MAX,MAX,MAX,MAX,MAX,0  ]],

    [[0  ,MAX,MAX,MAX,MAX,MAX],
     [MAX,0  ,4  ,4  ,MAX,MAX],
     [MAX,4  ,0  ,MAX,4  ,MAX],
     [MAX,4  ,MAX,0  ,4  ,MAX],
     [MAX,MAX,4  ,4  ,0  ,MAX],
     [MAX,MAX,MAX,MAX,MAX,0  ]],

    [[0  ,MAX,MAX,MAX,MAX,MAX],
     [MAX,0  ,4  ,4  ,MAX,MAX],
     [MAX,4  ,0  ,MAX,4  ,MAX],
     [MAX,4  ,MAX,0  ,4  ,MAX],
     [MAX,MAX,4  ,4  ,0  ,MAX],
     [MAX,MAX,MAX,MAX,MAX,0  ]],

    [[0  ,MAX,MAX,MAX,MAX,MAX],
     [MAX,0  ,4  ,4  ,MAX,MAX],
     [MAX,4  ,0  ,MAX,4  ,MAX],
     [MAX,4  ,MAX,0  ,4  ,MAX],
     [MAX,MAX,4  ,4  ,0  ,MAX],
     [MAX,MAX,MAX,MAX,MAX,0  ]],

    [[0  ,MAX,MAX,MAX,MAX,MAX],
     [MAX,0  ,4  ,4  ,MAX,MAX],
     [MAX,4  ,0  ,MAX,4  ,MAX],
     [MAX,4  ,MAX,0  ,4  ,MAX],
     [MAX,MAX,4  ,4  ,0  ,MAX],
     [MAX,MAX,MAX,MAX,MAX,0  ]],

    [[0  ,MAX,MAX,MAX,MAX,MAX],
     [MAX,0  ,4  ,4  ,MAX,MAX],
     [MAX,4  ,0  ,MAX,4  ,MAX],
     [MAX,4  ,MAX,0  ,4  ,MAX],
     [MAX,MAX,4  ,4  ,0  ,MAX],
     [MAX,MAX,MAX,MAX,MAX,0  ]]
    ] #각 목적지가 가지는 위치 행렬

flag = [0,0,0,0,0,0] #dijkstra 알고리즘을 위한 배열
dist = [
    [MAX,MAX,MAX,MAX,MAX,MAX],
    [MAX,MAX,MAX,MAX,MAX,MAX],
    [MAX,MAX,MAX,MAX,MAX,MAX],
    [MAX,MAX,MAX,MAX,MAX,MAX],
    [MAX,MAX,MAX,MAX,MAX,MAX],
    [MAX,MAX,MAX,MAX,MAX,MAX],
    [MAX,MAX,MAX,MAX,MAX,MAX],
    [MAX,MAX,MAX,MAX,MAX,MAX],
    [MAX,MAX,MAX,MAX,MAX,MAX],
    [MAX,MAX,MAX,MAX,MAX,MAX],
    [MAX,MAX,MAX,MAX,MAX,MAX],
    [MAX,MAX,MAX,MAX,MAX,MAX],
    [MAX,MAX,MAX,MAX,MAX,MAX],
    [MAX,MAX,MAX,MAX,MAX,MAX],
    [MAX,MAX,MAX,MAX,MAX,MAX]
    ] #차량으로부터의 거리를 나타내는 2차원 배열 (차량,(1,0),(1,4),(5,0),(5,4),목적지)
load = [
    [1,1,1,1,1,1],
    [1,1,1,1,1,1],
    [1,1,1,1,1,1],
    [1,1,1,1,1,1],
    [1,1,1,1,1,1],
    [1,1,1,1,1,1],
    [1,1,1,1,1,1],
    [1,1,1,1,1,1],
    [1,1,1,1,1,1],
    [1,1,1,1,1,1],
    [1,1,1,1,1,1],
    [1,1,1,1,1,1],
    [1,1,1,1,1,1],
    [1,1,1,1,1,1],
    [1,1,1,1,1,1]
    ] #경로를 나타내는 배열

direction = [0,0,0,1] #차량의 방향(동서남북)

direct_park= [
    [3,3,3,3,9,9],
    [9,9,9,9,9,0],
    [9,9,9,9,9,0],
    [9,9,9,9,9,0],
    [9,9,9,9,9,0],
    [9,9,9,9,9,9],
    [9,2,2,2,9,9],
    [9,9,9,9,9,9],
    [9,9,9,9,9,9],
    [9,9,9,9,9,9],
    ]

def car_position(): #차량 위치 찾는 함수
    global x
    global y
    global parkinglot
    i=0
    while i<10:
        j=0
        while j<6:
            if parkinglot[i][j] == 4:
                x=i
                y=j
            j = j+1
        i = i+1
    return;

def des_position(): #목적지 위치 찾는 함수
    global pos_num
    global X
    global Y
    global parkinglot
    i=0
    pos_num = 0
    while i<10:
        j=0
        while j<6:
            if parkinglot[i][j] == 2:
                X[pos_num] = i
                Y[pos_num] = j
                pos_num = pos_num + 1
            j = j + 1
        i = i + 1
    return;

def initiate_des_node(): #목적지 위치 초기화 함수
    global des_node
    i=0
    while i<15:
        j=0
        while j<3:
            des_node[i][j] = 0
            j = j + 1
        i = i + 1
    return;

def trans_node(): #목적지까지 가려면 어디로 가야하는지 찾는 함수
    global node_num
    global pos_num
    global X
    global Y
    global des_node
    i=0
    node_num = 0
    initiate_des_node()
    while i<pos_num:
        if Y[i] != 5:
            if X[i] == 0 or X[i] == 4:
                des_node[node_num][0] = X[i] + 1
                des_node[node_num][1] = Y[i]
                des_node[node_num][2] = i + 1
                node_num = node_num + 1
            else:
                des_node[node_num][0] = X[i] - 1
                des_node[node_num][1] = Y[i]
                des_node[node_num][2] = i + 1
                node_num = node_num + 1
        else:
           des_node[node_num][0] = X[i]
           des_node[node_num][1] = Y[i] - 1
           des_node[node_num][2] = i + 1
           node_num = node_num + 1
        i = i + 1
    return;

def initiate_tracking_node(): #거리 행렬 초기화 함수
    global tracking_node
    i=0
    init = [
     [0  ,MAX,MAX,MAX,MAX,MAX],
     [MAX,0  ,4  ,4  ,MAX,MAX],
     [MAX,4  ,0  ,MAX,4  ,MAX],
     [MAX,4  ,MAX,0  ,4  ,MAX],
     [MAX,MAX,4  ,4  ,0  ,MAX],
     [MAX,MAX,MAX,MAX,MAX,0  ]
     ]
    while i<15:
        j=0
        while j<6:
            k=0
            while k<6:
                tracking_node[i][j][k] = init[j][k]
                k = k + 1
            j = j + 1
        i = i + 1
    return;

def tracking_start(): #거리행렬에 차량 위치 넣는 함수
    global x
    global y
    global node_num
    global tracking_node
    global direction
    i=0
    initiate_tracking_node()
    while i < node_num:
        if x == 1 and y == 0:
            tracking_node[i][0][1] = 0
            tracking_node[i][1][0] = 0
            if direction[3] == 1:
                tracking_node[i][1][3] = MAX
                tracking_node[i][3][1] = MAX
            elif direction[1] == 1:
                tracking_node[i][1][2] = MAX
                tracking_node[i][2][1] = MAX
        elif x == 1 and y == 4:
            tracking_node[i][0][2] = 0
            tracking_node[i][2][0] = 0
            if direction[3] == 1:
                tracking_node[i][2][4] = MAX
                tracking_node[i][4][2] = MAX
            elif direction[0] == 1:
                tracking_node[i][2][1] = MAX
                tracking_node[i][1][2] = MAX
        elif x == 5 and y == 0:
            tracking_node[i][0][3] = 0
            tracking_node[i][3][0] = 0
            if direction[1] == 1:
                tracking_node[i][3][4] = MAX
                tracking_node[i][4][3] = MAX
            elif direction[2] == 1:
                tracking_node[i][3][1] = MAX
                tracking_node[i][1][3] = MAX
        elif x == 5 and y == 4:
            tracking_node[i][0][4] = 0
            tracking_node[i][4][0] = 0
            if direction[0] == 1:
                tracking_node[i][4][3] = MAX
                tracking_node[i][3][4] = MAX
            elif direction[2] == 1:
                tracking_node[i][4][2] = MAX
                tracking_node[i][2][4] = MAX
        elif x > 5:
            tracking_node[i][0][3] = x - 5
            tracking_node[i][3][0] = x - 5
            direction[0] = 0
            direction[1] = 0
            direction[2] = 0
            direction[3] = 1
        else:
            if x == 1:
                tracking_node[i][0][1] = y
                tracking_node[i][1][0] = y
                tracking_node[i][0][2] = 4 - y
                tracking_node[i][2][0] = 4 - y
                tracking_node[i][1][2] = MAX
                tracking_node[i][2][1] = MAX
                if direction[1] == 1:
                    tracking_node[i][2][0] = MAX
                    tracking_node[i][0][2] = MAX
                elif direction[0] == 1:
                    tracking_node[i][1][0] = MAX
                    tracking_node[i][0][1] = MAX
            elif x == 5:
                tracking_node[i][0][3] = y
                tracking_node[i][3][0] = y
                tracking_node[i][0][4] = 4 - y
                tracking_node[i][4][0] = 4 - y
                tracking_node[i][3][4] = MAX
                tracking_node[i][4][3] = MAX
                if direction[1] == 1:
                    tracking_node[i][4][0] = MAX
                    tracking_node[i][0][4] = MAX
                elif direction[0] == 1:
                    tracking_node[i][3][0] = MAX
                    tracking_node[i][0][3] = MAX
            elif y == 0:
                tracking_node[i][0][1] = x - 1
                tracking_node[i][1][0] = x - 1
                tracking_node[i][0][3] = 5 - x
                tracking_node[i][3][0] = 5 - x
                tracking_node[i][1][3] = MAX
                tracking_node[i][3][1] = MAX
                if direction[3] == 1:
                    tracking_node[i][3][0] = MAX
                    tracking_node[i][0][3] = MAX
                elif direction[2] == 1:
                    tracking_node[i][1][0] = MAX
                    tracking_node[i][0][1] = MAX
            elif y == 4:
                tracking_node[i][0][2] = x - 1
                tracking_node[i][2][0] = x - 1
                tracking_node[i][0][4] = 5 - x
                tracking_node[i][4][0] = 5 - x
                tracking_node[i][2][4] = MAX
                tracking_node[i][4][2] = MAX
                if direction[3] == 1:
                    tracking_node[i][4][0] = MAX
                    tracking_node[i][0][4] = MAX
                elif direction[2] == 1:
                    tracking_node[i][2][0] = MAX
                    tracking_node[i][0][2] = MAX
        i = i+1
    return;

def tracking_des(): #거리 행렬에 목적지 위치 넣는 함수
    global node_num
    global des_node
    global tracking_node
    global x
    global y
    global direction
    i = 0
    while i < node_num:
        if des_node[i][0] == 1:
            tracking_node[i][1][5] = des_node[i][1]
            tracking_node[i][5][1] = des_node[i][1]
            tracking_node[i][2][5] = 4 - des_node[i][1]
            tracking_node[i][5][2] = 4 - des_node[i][1]
            if x == 1:
                if direction[1] == 1:
                    if y >= des_node[i][1]:
                        tracking_node[i][0][5] = y - des_node[i][1]
                        tracking_node[i][5][0] = y - des_node[i][1]
                    else:
                        tracking_node[i][1][5] = MAX
                        tracking_node[i][5][1] = MAX
                elif direction[0] == 1:
                    if y <= des_node[i][1]:
                        tracking_node[i][0][5] = des_node[i][1] - y
                        tracking_node[i][5][0] = des_node[i][1] - y
                    else:
                        tracking_node[i][2][5] = MAX
                        tracking_node[i][5][2] = MAX
        elif des_node[i][0] == 5:
            tracking_node[i][3][5] = des_node[i][1]
            tracking_node[i][5][3] = des_node[i][1]
            tracking_node[i][4][5] = 4 - des_node[i][1]
            tracking_node[i][5][4] = 4 - des_node[i][1]
            if x == 5:
                if direction[1] == 1:
                    if y >= des_node[i][1]:
                        tracking_node[i][0][5] = y - des_node[i][1]
                        tracking_node[i][5][0] = y - des_node[i][1]
                    else:
                        tracking_node[i][3][5] = MAX
                        tracking_node[i][5][3] = MAX
                elif direction[0] == 1:
                    if y <= des_node[i][1]:
                        tracking_node[i][0][5] = des_node[i][1] - y
                        tracking_node[i][5][0] = des_node[i][1] - y
                    else:
                        tracking_node[i][4][5] = MAX
                        tracking_node[i][5][4] = MAX
        elif des_node[i][1] == 4:
            tracking_node[i][2][5] = des_node[i][0] - 1
            tracking_node[i][5][2] = des_node[i][0] - 1
            tracking_node[i][4][5] = 5 - des_node[i][0]
            tracking_node[i][5][4] = 5 - des_node[i][0]
            if y == 4:
                if direction[3] == 1:
                    if x >= des_node[i][0]:
                        tracking_node[i][0][5] = x - des_node[i][0]
                        tracking_node[i][5][0] = x - des_node[i][0]
                    else:
                        tracking_node[i][2][5] = MAX
                        tracking_node[i][5][2] = MAX
                elif direction[2] == 1:
                    if x <= des_node[i][0]:
                        tracking_node[i][0][5] = des_node[i][0] - x
                        tracking_node[i][5][0] = des_node[i][0] - x
                    else:
                        tracking_node[i][4][5] = MAX
                        tracking_node[i][5][4] = MAX
        i = i+1
    return;

def dijkstra(): #최단거리와 경로를 찾는 함수
    global node_num
    global flag
    global dist
    global load
    global tracking_node
    k = 0
    temp = 0
    MIN = MAX
    while k<node_num:
        i = 0
        while i < 6:
            flag[i] = 0
            dist[k][i] = MAX
            load[k][i] = 1
            i = i + 1
        dist[k][0] = 0
        i = 0
        while i < 6:
            MIN = MAX
            j = 0
            while j < 6:
                if MIN > dist[k][j] and flag[j] == 0:
                    MIN = dist[k][j]
                    temp = j
                j = j + 1
            flag[temp] = 1
            j = 0
            while j < 6:
                if dist[k][j] > tracking_node[k][temp][j] + dist[k][temp] and tracking_node[k][temp][j] != MAX:
                    dist[k][j] = tracking_node[k][temp][j] + dist[k][temp]
                    load[k][j] = load[k][temp] * 10 + j + 1
                j = j + 1
            i = i + 1
        k = k + 1
    return;

def tracking(dest): #경로에 따라 불을 켜는 함수
    global load
    global parkinglot
    global x
    global y
    global des_node
    node_a = 1
    node_b = 1
    j = 0
    temp = load[dest][5]
    while temp >10 :
        node_a = temp % 10
        temp = int(temp / 10)
        node_b = temp % 10
        if node_a != 6:
            if node_a == 2:
                if node_b == 3:
                    parkinglot[1][0] = 1
                    parkinglot[1][1] = 1
                    parkinglot[1][2] = 1
                    parkinglot[1][3] = 1
                    parkinglot[1][4] = 1
                elif node_b == 4:
                    parkinglot[1][0] = 1
                    parkinglot[2][0] = 1
                    parkinglot[3][0] = 1
                    parkinglot[4][0] = 1
                    parkinglot[5][0] = 1
                else:
                    if x == 1:
                        j = 0
                        while j < y:
                            parkinglot[1][j] = 1
                            j = j + 1
                    else :
                        j = 1
                        while j < x:
                            parkinglot[j][0] = 1
                            j = j + 1
            elif node_a == 3:
                if node_b == 2:
                    parkinglot[1][0] = 1
                    parkinglot[1][1] = 1
                    parkinglot[1][2] = 1
                    parkinglot[1][3] = 1
                    parkinglot[1][4] = 1
                elif node_b == 5:
                    parkinglot[1][4] = 1
                    parkinglot[2][4] = 1
                    parkinglot[3][4] = 1
                    parkinglot[4][4] = 1
                    parkinglot[5][4] = 1
                else:
                    if x == 1:
                        j = 4
                        while j > y:
                            parkinglot[1][j] = 1
                            j = j - 1
                    else :
                        j = 1
                        while j < x:
                            parkinglot[j][4] = 1
                            j = j + 1
            elif node_a == 4:
                if node_b == 2:
                    parkinglot[1][0] = 1
                    parkinglot[2][0] = 1
                    parkinglot[3][0] = 1
                    parkinglot[4][0] = 1
                    parkinglot[5][0] = 1
                elif node_b == 5:
                    parkinglot[5][0] = 1
                    parkinglot[5][1] = 1
                    parkinglot[5][2] = 1
                    parkinglot[5][3] = 1
                    parkinglot[5][4] = 1
                else:
                    if x == 5:
                        j = 0
                        while j < y:
                            parkinglot[5][j] = 1
                            j = j + 1
                    elif x > 5:
                        j = 5
                        while j < x:
                            parkinglot[j][0] = 1
                            j = j + 1
                    else :
                        j = 5
                        while j > x:
                            parkinglot[j][0] = 1
                            j = j - 1
            elif node_a == 5:
                if node_b == 3:
                    parkinglot[1][4] = 1
                    parkinglot[2][4] = 1
                    parkinglot[3][4] = 1
                    parkinglot[4][4] = 1
                    parkinglot[5][4] = 1
                elif node_b == 4:
                    parkinglot[5][0] = 1
                    parkinglot[5][1] = 1
                    parkinglot[5][2] = 1
                    parkinglot[5][3] = 1
                    parkinglot[5][4] = 1
                else:
                    if x == 5:
                        j = 4
                        while j > y:
                            parkinglot[5][j] = 1
                            j = j - 1
                    else :
                        j = 5
                        while j > x:
                            parkinglot[j][4] = 1
                            j = j - 1
            if parkinglot[x][y] == 1:
                parkinglot[x][y] = 4
        else:
            if node_b == 1:
                if x == des_node[dest][0]:
                    if y < des_node[dest][1]:
                        j = des_node[dest][1]
                        while j > y:
                            parkinglot[x][j] = 1
                            j = j - 1
                    else:
                        j = des_node[dest][1]
                        while j < y:
                            parkinglot[x][j] = 1
                            j = j + 1
                else:
                    if x < des_node[dest][0]:
                        j = des_node[dest][0]
                        while j > x:
                            parkinglot[j][y] = 1
                            j = j - 1
                    else:
                        j = des_node[dest][0]
                        while j < x:
                            parkinglot[j][y] = 1
                            j = j + 1
            else:
                if node_b == 2:
                    if des_node[dest][0] == 1:
                        j = 0
                        while j <= des_node[dest][1]:
                            parkinglot[1][j] = 1
                            j = j + 1
                    else:
                        j = 1
                        while j <= des_node[dest][0]:
                            parkinglot[j][0] = 1
                            j = j + 1
                elif node_b == 3:
                    if des_node[dest][0] == 1:
                        j = 4
                        while j >= des_node[dest][1]:
                            parkinglot[1][j] = 1
                            j = j - 1
                    else:
                        j = 1
                        while j <= des_node[dest][0]:
                            parkinglot[j][4] = 1
                            j = j + 1
                elif node_b == 4:
                    if des_node[dest][0] == 5:
                        j = 0
                        while j <= des_node[dest][1]:
                            parkinglot[5][j] = 1
                            j = j + 1
                    else:
                        j = 5
                        while j >= des_node[dest][0]:
                            parkinglot[j][0] = 1
                            j = j - 1
                else:
                    if des_node[dest][0] == 5:
                        j = 4
                        while j >= des_node[dest][1]:
                            parkinglot[5][j] = 1
                            j = j - 1
                    else:
                        j = 4
                        while j >= des_node[dest][0]:
                            parkinglot[j][4] = 1
                            j = j - 1
            if x != des_node[dest][0] or y != des_node[dest][1]:
                parkinglot[des_node[dest][0]][des_node[dest][1]] = 6
        if parkinglot[des_node[dest][0]][des_node[dest][1]] == 1:
            parkinglot[des_node[dest][0]][des_node[dest][1]] = 6
    return;

def initiate_led(): #LED초기화 함수
    global parkinglot
    i = 0
    while i < 10:
        j = 0
        while j < 5:
            if parkinglot[i][j] == 1 or parkinglot[i][j] == 6:
                parkinglot[i][j] = 0
            j = j + 1
        i = i + 1
    return;

def update_tracking(): #경로를 벗어났을때 새로 경로 찾는 함수
    global destination
    destination = 0
    initiate_led()
    car_position()
    des_position()
    trans_node()
    tracking_start()
    tracking_des()
    dijkstra()
    i = 0
    MIN = MAX
    while i < node_num:
        if MIN > dist[i][5]:
            MIN = dist[i][5]
            destination = i
        i = i + 1
    tracking(destination)
    return;

def showMap(): #주차장 데이터를 출력하는 함수
    global parkinglot
    i=0
    display = [
        ["○","○","○","○","○","○"],
        ["○","○","○","○","○","○"],
        ["○","○","○","○","○","○"],
        ["○","○","○","○","○","○"],
        ["○","○","○","○","○","○"],
        ["○","○","○","○","○","○"],
        ["○","○","○","○","○","○"],
        ["○","○","○","○","○","○"],
        ["○","○","○","○","○","○"],
        ["○","○","○","○","○","○"]
        ]
    while i < 10:
        j = 0
        while j < 6:
            if parkinglot[i][j] == 0:
                display[i][j] = "○" #꺼진 led
            elif parkinglot[i][j] == 1:
                display[i][j] = "●" #켜진 led
            elif parkinglot[i][j] == 2:
                display[i][j] = "□" #빈자리
            elif parkinglot[i][j] == 3:
                display[i][j] = "■" #주차된 자리
            elif parkinglot[i][j] == 4:
                display[i][j] = "★" #차량
            elif parkinglot[i][j] == 6:
                display[i][j] = "☆" #도착지점을 알려주는 지점
            else:
                display[i][j] = "※"
            j = j + 1
        i = i + 1
    i = 0
    while i < 10:
        print(display[i])
        i = i + 1
    return;

def makingMassenger(): #주차장 데이터를 출력하는 함수
    global parkinglot
    global Massenger
    global isCarbeing
    i = 0
    msg = ''
    while i < 10:
        j = 0
        while j < 6:
            if parkinglot[i][j] == 0:
                msg = msg + str(i)+':'+str(j)+':'+'0'+':'+'1'+','
            elif parkinglot[i][j] == 1:
                msg = msg + str(i)+':'+str(j)+':'+'1'+':'+'0'+','
            elif parkinglot[i][j] == 4:
                msg = msg + str(i)+':'+str(j)+':'+'0'+':'+'1'+','
                isCarbeing = True
            elif parkinglot[i][j] == 6:
                msg = msg + str(i)+':'+str(j)+':'+'2'+':'+'0'+','
            j = j + 1
        i = i + 1
    i = 0
    Massenger = msg
    return isCarbeing;


def print_parking_lot(): #주차장 데이터를 출력하는 함수
    global parkinglot
    global Massenger
    global preData
    global INITMSG
    carinit ='9:5:0:0'
    initCar = ['9','5','0','0']
    i = 0
    isCarbeing = False
    remains = 0
    msg = ''
    display = [
        ["○","○","○","○","○","○"],
        ["○","○","○","○","○","○"],
        ["○","○","○","○","○","○"],
        ["○","○","○","○","○","○"],
        ["○","○","○","○","○","○"],
        ["○","○","○","○","○","○"],
        ["○","○","○","○","○","○"],
        ["○","○","○","○","○","○"],
        ["○","○","○","○","○","○"],
        ["○","○","○","○","○","○"]
        ]
    while i < 10:
        j = 0
        while j < 6:
            if parkinglot[i][j] == 0:
                display[i][j] = "○" #꺼진 led
                msg = msg + str(i)+':'+str(j)+':'+'0'+':'+'1'+','
            elif parkinglot[i][j] == 1:
                display[i][j] = "●" #켜진 led
                msg = msg + str(i)+':'+str(j)+':'+'1'+':'+'0'+','
            elif parkinglot[i][j] == 2:
                display[i][j] = "□" #빈자리
                remains = remains + 100
            elif parkinglot[i][j] == 3:
                display[i][j] = "■" #주차된 자리
                remains = remains + 1
            elif parkinglot[i][j] == 4:
                display[i][j] = "★" #차량
                msg = msg + str(i)+':'+str(j)+':'+'0'+':'+'1'+','
                carinit = str(i) + ':' + str(j) + ':0:0'
                isCarbeing = True
            elif parkinglot[i][j] == 6:
                display[i][j] = "☆" #도착지점을 알려주는 지점
                msg = msg + str(i)+':'+str(j)+':'+'2'+':'+'0'+','
                remains = remains + 10000
            else:
                display[i][j] = "※"
            j = j + 1
        i = i + 1
    i = 0
    Massenger = msg
    print(remains)
    if remains != 10010:
        while i < 10:
            print(display[i])
            i = i + 1
        return isCarbeing;
    else:
        Massenger = INITMSG
        initCar = carinit.split(':')
        updateDB(initCar)
        preData = '9:0:0:0'
        return True;
    

def main():
    global parkinglot
    global x
    global y
    global direction
    global destination
    isCarbeing = False
    roop = 1
    if roop == 1:
        destination = 0
        initiate_led()
        car_position()
        des_position()
        trans_node()
        #if pos_num == 0:
        #   print("빈 자리가 없습니다.")
        #   roop = 0
        if True:
            tracking_start()
            tracking_des()
            dijkstra()
            i = 0
            MIN = MAX
            while i < node_num:
                if MIN > dist[i][5]:
                    MIN = dist[i][5]
                    destination = i
                i = i + 1
            tracking(destination)
            #makingMassenger()
            print(" ")
            isCarbeing = print_parking_lot()
            parkinglot = parseMap()
            msgDivider()
            return isCarbeing
        
def assignTo1(sector):
    global MSG1
    if MSG1 == '':
        MSG1 = sector
    else:
        MSG1 = MSG1 + ',' + sector
    
def assignTo2(sector):
    global MSG2
    if MSG2 == '':
        MSG2 = sector
    else:
        MSG2 = MSG2 + ',' + sector
    
def assignTo3(sector):
    global MSG3
    if MSG3 == '':
        MSG3 = sector
    else:
        MSG3 = MSG3 + ',' + sector
    
def assignTo4(sector):
    global MSG4
    if MSG4 == '':
        MSG4 = sector
    else:
        MSG4 = MSG4 + ',' + sector
        
def assignTo5(sector):
    global MSG5
    if MSG5 == '':
        MSG5 = sector
    else:
        MSG5 = MSG5 + ',' + sector
        
def assignTo6(sector):
    global MSG6
    if MSG6 == '':
        MSG6 = sector
    else:
        MSG6 = MSG6 + ',' + sector
        
def assignTo7(sector):
    global MSG7
    if MSG7 == '':
        MSG7 = sector
    else:
        MSG7 = MSG7 + ',' + sector

def INITMSG():
    global MSG1
    global MSG2
    global MSG3
    global MSG4
    global MSG5
    global MSG6
    global MSG7
    MSG1 = ''
    MSG2 = ''
    MSG3 = ''
    MSG4 = ''
    MSG5 = ''
    MSG6 = ''
    MSG7 = ''

def identifyMsg(MAPSECTOR):
    INITMSG()
    for i in range(len(MAPSECTOR)):
        tmpSector = MAPSECTOR[i]
        MAPDATA = MAPSECTOR[i].split(':')
        if MAPDATA[0] == '5' and MAPDATA[1] == '4':### pi 1
            assignTo1(tmpSector)
        elif MAPDATA[0] == '4' and MAPDATA[1] == '4':
            assignTo1(tmpSector)
        elif MAPDATA[0] == '3' and MAPDATA[1] == '4':
            assignTo1(tmpSector)
        elif MAPDATA[0] == '2' and MAPDATA[1] == '4':
            assignTo1(tmpSector)
        elif MAPDATA[0] == '1' and MAPDATA[1] == '4':
            assignTo1(tmpSector)
        if MAPDATA[0] == '1' and MAPDATA[1] == '4':### pi 2
            assignTo2(tmpSector)
        elif MAPDATA[0] == '1' and MAPDATA[1] == '3':
            assignTo2(tmpSector)
        elif MAPDATA[0] == '1' and MAPDATA[1] == '2':
            assignTo2(tmpSector)
        elif MAPDATA[0] == '1' and MAPDATA[1] == '1':
            assignTo2(tmpSector)
        elif MAPDATA[0] == '1' and MAPDATA[1] == '0':
            assignTo2(tmpSector)
        if MAPDATA[0] == '1' and MAPDATA[1] == '0':### pi 3
            assignTo3(tmpSector)
        elif MAPDATA[0] == '2' and MAPDATA[1] == '0':
            assignTo3(tmpSector)
        elif MAPDATA[0] == '3' and MAPDATA[1] == '0':
            assignTo3(tmpSector)
        elif MAPDATA[0] == '4' and MAPDATA[1] == '0':
            assignTo3(tmpSector)
        elif MAPDATA[0] == '5' and MAPDATA[1] == '0':
            assignTo3(tmpSector)
        if MAPDATA[0] == '5' and MAPDATA[1] == '0':### pi 4
            assignTo4(tmpSector)
        elif MAPDATA[0] == '6' and MAPDATA[1] == '0':
            assignTo4(tmpSector)
        elif MAPDATA[0] == '7' and MAPDATA[1] == '0':
            assignTo4(tmpSector)
        elif MAPDATA[0] == '8' and MAPDATA[1] == '0':
            assignTo4(tmpSector)
        elif MAPDATA[0] == '9' and MAPDATA[1] == '0':
            assignTo4(tmpSector)
        if MAPDATA[0] == '8' and MAPDATA[1] == '0':### pi 5 폐기 
            assignTo5(tmpSector)
        elif MAPDATA[0] == '8' and MAPDATA[1] == '1':
            assignTo5(tmpSector)
        elif MAPDATA[0] == '8' and MAPDATA[1] == '2':
            assignTo5(tmpSector)
        elif MAPDATA[0] == '8' and MAPDATA[1] == '3':
            assignTo5(tmpSector)
        elif MAPDATA[0] == '8' and MAPDATA[1] == '4':
            assignTo5(tmpSector)
        if MAPDATA[0] == '8' and MAPDATA[1] == '4':### pi 6 폐기
            assignTo6(tmpSector)
        elif MAPDATA[0] == '7' and MAPDATA[1] == '4':
            assignTo6(tmpSector)
        elif MAPDATA[0] == '6' and MAPDATA[1] == '4':
            assignTo6(tmpSector)
        elif MAPDATA[0] == '5' and MAPDATA[1] == '4':
            assignTo6(tmpSector)
        elif MAPDATA[0] == '4' and MAPDATA[1] == '4':
            assignTo6(tmpSector)
        if MAPDATA[0] == '5' and MAPDATA[1] == '4':### pi 7
            assignTo7(tmpSector)
        elif MAPDATA[0] == '5' and MAPDATA[1] == '3':
            assignTo7(tmpSector)
        elif MAPDATA[0] == '5' and MAPDATA[1] == '2':
            assignTo7(tmpSector)
        elif MAPDATA[0] == '5' and MAPDATA[1] == '1':
            assignTo7(tmpSector)
        elif MAPDATA[0] == '5' and MAPDATA[1] == '0':
            assignTo7(tmpSector)

def msgDivider():
    global Massenger
    tmpSector = ''
    MAPSECTOR = Massenger.split(',')
        identifyMsg(MAPSECTOR)
    Massenger = ''
    
    
    

def updateDB(MAPDATA):
    DBCONN = pymysql.connect(host = 'localhost',
                           user = 'root',
                           password = None,
                           db = 'car_map_data',
                           charset = 'utf8')
    with DBCONN.cursor() as cursor:
            sql = 'UPDATE map SET car = %s, parked = %s WHERE x = %s and y = %s'
            cursor.execute(sql, (MAPDATA[3], MAPDATA[2], MAPDATA[0], MAPDATA[1]))
            DBCONN.commit()
    DBCONN.close()
    
def Binding(port):
    s = socket.socket()
    host = ''
    s.bind((host,port))
    s.listen(5)
    return s

def calcDirection():
    global postX
    global postY
    global preX
    global preY
    global direction
    calcX = 0
    calcY = 0
    calcX = int(postX) - int(preX)
    calcY = int(postY) - int(preY)
    direct = [0,0,0,0]
    if calcX >= 1:
        direct = [0,0,1,0]#Direction = [1,0,0,0]
    elif calcX <= -1:
        direct = [0,0,0,1]#Direction = [0,1,0,0]
    elif calcY >= 1:
        direct = [1,0,0,0]#Direction = [0,0,0,1]
    elif calcY <= -1:
        direct = [0,1,0,0]#Direction = [0,0,1,0]
    else:
        direct = direction#Direction = [0,0,0,1]
    direction = direct
    
def splitDatasforInit(MAPSECTOR):
    for i in range(0,(len(MAPSECTOR))):
            MAPDATA = MAPSECTOR[i].split(':')
            updateDB(MAPDATA)

def splitDatasforParking(MAPSECTOR):
    for i in range(0,(len(MAPSECTOR)-1)):
            MAPDATA = MAPSECTOR[i].split(':')
            updateDB(MAPDATA)

def splitDatas(MAPSECTOR):
        global postX
        global postY
        MAPDATA = MAPSECTOR.split(':')
        if MAPDATA[0] == '0' and MAPDATA[1] == '0':
            print('no')
            return False # 더미값 받음
        else: 
            postX = MAPDATA[0]
            postY = MAPDATA[1]
            calcDirection()
            print('gotit')
            updateDB(MAPDATA)
            return True # 제대로 된 값을 받음
        
        
def splitDatasforPre(MAPSECTOR):
        global preX
        global preY
        MAPDATA = MAPSECTOR.split(':')
        preX = MAPDATA[0]
        preY = MAPDATA[1]
        MAPDATA[2] = '0'
        MAPDATA[3] = '0'
        updateDB(MAPDATA)
    
def assignToID(ID):
    global MSG1, MSG2, MSG3, MSG4, MSG5, MSG6, MSG7
    data = ''
    if ID == '1':
        data = MSG1
    elif ID == '2':
        data = MSG2
    elif ID == '3':
        data = MSG3
    elif ID == '4':
        data = MSG4
    elif ID == '5':
        data = MSG5
    elif ID == '6':
        data = MSG6
    elif ID == '7':
        data = MSG7
    return data

def sendData(c, addr, ID):
    data = assignToID(ID)
    print(data)
    byte_data = str.encode(data)
    c.sendall(byte_data)
    c.close()
    
    
def read(s):
    try:
        msg = s.recv(1024)
        #s.close()
    except socket.error as msg:
        sys.stderr.write('error %s' %msg[1])
        #s.close()
        print('close')
        sys.exit(2)
    return msg

def accuracyTest(preData, strID):
    MAPDATA = [11, 11]
    strMAPDATA = preData.split(':')
    MAPDATA[0] = int(strMAPDATA[0])
    MAPDATA[1] = int(strMAPDATA[1])
    ID = int(strID)
    if MAPDATA[0] == 1 and MAPDATA [1] == 0:
        if ID == 2 or ID == 3:
            return True
        else:
            return False
    elif MAPDATA[0] == 1 and MAPDATA [1] == 4:
        if ID == 1 or ID == 2:
            return True
        else:
            return False
    elif MAPDATA[0] == 5 and MAPDATA[1] == 0:
        if ID == 3 or ID == 4 or ID ==7:
            return True
        else:
            return False
    elif MAPDATA[0] == 5 and MAPDATA[1] == 4:
        if ID == 1 or ID == 7:
            return True
        else:
            return False
    elif (MAPDATA[0] >= 5 and MAPDATA[0] <= 9) and MAPDATA[1] == 0:
        if ID == 4:
            return True
        else:
            return False
    elif (MAPDATA[0] >= 1 and MAPDATA[0] <= 5) and MAPDATA[1] == 0:
        if ID == 3:
            return True
        else:
            return False
    elif MAPDATA[0] == 5 and (MAPDATA[1] >= 0 and MAPDATA[1] <= 4):
        if ID == 7:
            return True
        else:
            return False
    elif MAPDATA[0] ==1 and (MAPDATA[1] >= 0 and MAPDATA[1] <= 4):
        if ID == 2:
            return True
        else:
            return False
    elif (MAPDATA[0] >= 1 and MAPDATA[0] <= 5) and MAPDATA[1] == 4:
        if ID == 1:
            return True
        else:
            return False
    else:
        return False

def readData(s, isCarbeing):
        global preData
        tempValue = False
        data = read(s).decode('utf-8')
        print('received data : %s' % data)
        MAPDATA = data.split(',')
        if len(MAPDATA) == 4:
            splitDatasforParking(MAPDATA)
            splitDatasforPre(preData)
            return MAPDATA[1]
        
        elif len(MAPDATA) ==2:
            if accuracyTest(preData, MAPDATA[1]):
                tempValue = splitDatas(MAPDATA[0])
                if tempValue:
                    splitDatasforPre(preData)
                    preData = data
                    print(preData)
                else:
                    print(preData)
                return MAPDATA[1] ## return ID
            
            else:
                print('data is received but ignore it')
                return MAPDATA[1]
        return MAPDATA[1]
    
    
            
               
        
        
def comm(s, port):
    global direction
    global preData
    isCarbeing =False
    ID = ''
    c, addr = s.accept()
    isCarbeing = main()
    ID = readData(c, isCarbeing)
    sendData(c, addr, ID)
    #showMap()
    print(preData)
    
    

def server():
    if __name__ == '__main__':
        port = 12346
        global INITMAP
        splitDatasforInit(INITMAP.split(','))
        s = Binding(port)
        #s2 = Binding(port2)
        #s3 = Binding(port3)
        while True:
            comm(s, port)
            #showMap()
            #comm(s2, port2)
            #comm(s3, port3)
           
            
server()
