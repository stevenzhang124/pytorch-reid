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

p1 = "Person_1"
p2 = "Person_2"
c1 = 116
c2 = 106
c3 = 117
c4 = 113
identify_name = "Person_113"

cameras = [106,113,115,116,117,118]
pos_x = [264,264,274,431,265]
pos_y = [480,480,407,319,298]


for i in range(len(cameras)):
	if cameras[i] == 117:
		pos = [[557,100],[387,308],[400,455],[470,470],[20,468],[620,475],[410,475],[215,470]]
		for j in range(8):
			sql1 = "insert into REID(person, ctime, camera, pos_x, pos_y) values ('{0}',{1},{2},{3},{4})".format("Person_"+str(cameras[i]), int(time.time()), cameras[i], pos[j][0], pos[j][1])
			print(sql1)
			cur.execute(sql1)
		continue
	for j in range(5):
		sql1 = "insert into REID(person, ctime, camera, pos_x, pos_y) values ('{0}',{1},{2},{3},{4})".format("Person_"+str(cameras[i]), int(time.time()), cameras[i], pos_x[j], pos_y[j])
		print(sql1)
		cur.execute(sql1)
	time.sleep(0.1)

# sql4 = "insert into REID(person,ctime, camera) values ('{0}',{1},{2})".format(str(p2), int(time.time()-1000000), 116)
# print(sql4)
# cur.execute(sql4)




cur.execute("select * from REID ")

datas = cur.fetchall()
for data in datas:
	print(data)

#cur.execute("DROP TABLE REID")
mydb.close()	
