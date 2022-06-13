import mysql.connector
import time

def db_connection():
    mydb = mysql.connector.connect(host='192.168.1.103',  # 192.168.1.100
                                   user='cqy',
                                   port='3306',
                                   database='edgedemo',
                                   passwd='123456',
                                   autocommit=True)
    return mydb

mydb = db_connection()
cur = mydb.cursor()
print("Opened database successfully")

cameras = [106,113,115,116,117,118]
gender, age, hair, luggage, attire = "male", "young", 'long', 'NA', 'drip'

pos = [
    [[100,435],[420,475],[600,470],[345,255],[320,150]],
    [[585,475],[420,470],[280,475],[280,250],[280,175]],
    [[600,290],[320,290],[30,290],[300,350],[300,450]],
    [[80,360],[400,375],[580,400],[525,475],[520,320]],
    [[70,320],[260,330],[440,350],[440,475],[620,475]],
    [[160,475],[620,475],[400,285],[330,240],[160,225]]
]
for i in range(len(pos)):
    for j in range(len(pos[i])):
        sql1 = "INSERT into MOT(person, ctime, camera, pos_x, pos_y, gender, age, hair, luggage, attire ) values ('{0}', {1}, {2}, {3}, {4}, '{5}', '{6}', '{7}', '{8}', '{9}')".format("Person_"+str(cameras[i]), int(time.time()), cameras[i], pos[i][j][0], pos[i][j][1], gender, age, hair, luggage, attire)
        print(sql1)
        cur.execute(sql1)
    time.sleep(0.1)

cur.execute("select * from MOT ")

datas = cur.fetchall()
for data in datas:
    print(data)

#cur.execute("DROP TABLE MOT")
mydb.close()    

