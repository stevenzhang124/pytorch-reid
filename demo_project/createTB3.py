import mysql.connector

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

cur.execute("DROP TABLE IF EXISTS REID")
cur.execute("DROP TABLE IF EXISTS MOT")


print("Delete table successfully")

cur.execute('''CREATE TABLE IF NOT EXISTS MOT
		(
		id int(11) unsigned NOT NULL AUTO_INCREMENT,
		person varchar(40) NOT NULL,
		ctime bigint(11) NOT NULL,
		camera int(11) NOT NULL,
		pos_x int(11) NOT NULL,
		pos_y int(11) NOT NULL,
		gender varchar(40) DEFAULT "N/A" NOT NULL,
		age varchar(40) DEFAULT "N/A" NOT NULL,
		hair varchar(40) DEFAULT "N/A" NOT NULL,
		luggage varchar(40) DEFAULT "N/A" NOT NULL,
		attire varchar(40) DEFAULT "N/A" NOT NULL,
		PRIMARY KEY (id)
		) ENGINE=InnoDB AUTO_INCREMENT=0 DEFAULT CHARSET=utf8;''')

print("Table created successfully")

mydb.commit()
mydb.close()

