import time
# import mysql.connector
import json



class Database:
	"""
    Uses a shared database between edge
    ----------
    database : connection
        Shared Database
    cursor: cursor
    	Cursor
    camera: int
    	Current camera
    """

	def __init__(self, camera, database, num_cams):
		self.database = database
		self.cursor = self.database.cursor()
		self.camera = camera
		self.num_cams = num_cams
		# self.init()


	def init(self):
		self.cursor.execute("DROP TABLE IF EXISTS MOT")
		self.cursor.execute("DROP TABLE IF EXISTS TRACKS")
		self.cursor.execute("DROP TABLE IF EXISTS NEXTID")
		self.cursor.execute("DROP TABLE IF EXISTS DUPL_COUNT")
		self.cursor.execute("DROP TABLE IF EXISTS TOTAL")


		print("Deleted tables successfully")

		self.cursor.execute('''CREATE TABLE IF NOT EXISTS MOT
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
		attire varchar(100) DEFAULT "N/A" NOT NULL,
		PRIMARY KEY (id)
		) ENGINE=InnoDB AUTO_INCREMENT=0 DEFAULT CHARSET=utf8;''')

		# self.cursor.execute('''CREATE TABLE IF NOT EXISTS TRACKS
		# (
		# id int(11) unsigned NOT NULL AUTO_INCREMENT,
		# trk_id bigint(11) NOT NULL,
		# track LONGBLOB NOT NULL,
		# PRIMARY KEY (id)
		# ) ENGINE=InnoDB AUTO_INCREMENT=0 DEFAULT CHARSET=utf8;''')

		self.cursor.execute('''CREATE TABLE IF NOT EXISTS TRACKS
		(
		id int(11) unsigned NOT NULL AUTO_INCREMENT,
		trk_id bigint(11) NOT NULL,
		track LONGBLOB NOT NULL,
		is_duplicate int(1) DEFAULT 0 NOT NULL,
		PRIMARY KEY (id)
		) ENGINE=InnoDB AUTO_INCREMENT=0 DEFAULT CHARSET=utf8;''')

		self.cursor.execute('''CREATE TABLE IF NOT EXISTS NEXTID
		(
		id int(11) unsigned NOT NULL AUTO_INCREMENT,
		next_id int(11) NOT NULL,
		PRIMARY KEY (id)
		) ENGINE=InnoDB AUTO_INCREMENT=0 DEFAULT CHARSET=utf8;''')

		self.cursor.execute('''CREATE TABLE IF NOT EXISTS DUPL_COUNT
		(
		id int(11) unsigned NOT NULL AUTO_INCREMENT,
		origin_id int(1) NOT NULL,
		dupl_id int(1) NOT NULL,
		cam_id int(1) NOT NULL,
		PRIMARY KEY (id),
		UNIQUE KEY (cam_id, dupl_id)
		) ENGINE=InnoDB AUTO_INCREMENT=0 DEFAULT CHARSET=utf8;''')

		self.cursor.execute('''CREATE TABLE IF NOT EXISTS TOTAL
		(
		id int(11) unsigned NOT NULL AUTO_INCREMENT,
		entries int(11) NOT NULL,
		exits int(11) NOT NULL,
		PRIMARY KEY (id)
		) ENGINE=InnoDB AUTO_INCREMENT=0 DEFAULT CHARSET=utf8;''')


		print("Created Tables successfully")
		self.database.commit()


		# Initialize Next_ID
		self.cursor.execute("INSERT into NEXTID(next_id) values ({0})".format(1))
		self.cursor.execute("INSERT into TOTAL(entries, exits) values ({0},{1})".format(0,0))


	def insert_record(self, trk_id, pos_x, pos_y, attr_data):
		attr_data = attr_data[0]
		if attr_data != None:
			gender, age, hair, luggage, attire = attr_data[0], attr_data[1], attr_data[2], attr_data[3], attr_data[4] 
			info = "INSERT into MOT(person, ctime, camera, pos_x, pos_y, gender, age, hair, luggage, attire) values ('{0}', {1}, {2}, {3}, {4}, '{5}', '{6}', '{7}', '{8}', '{9}')".format(trk_id, int(time.time()), self.camera, pos_x, pos_y, gender, age, hair, luggage, attire)
			# print(info)
			self.cursor.execute(info)
		else:
			info = "SELECT * FROM MOT WHERE person = {0}".format(trk_id)
			self.cursor.execute(info)
			data = self.cursor.fetchall()
			if len(data) > 0:
				data = data[0]
				gender, age, hair, luggage, attire = data[6], data[7], data[8], data[9], data[10]
				info = "INSERT into MOT(person, ctime, camera, pos_x, pos_y, gender, age, hair, luggage, attire) values ('{0}', {1}, {2}, {3}, {4}, '{5}', '{6}', '{7}', '{8}', '{9}')".format(trk_id, int(time.time()), self.camera, pos_x, pos_y, gender, age, hair, luggage, attire)
				# print(info)
				self.cursor.execute(info)
			else:
				info = "INSERT into MOT(person, ctime, camera, pos_x, pos_y) values ('{0}', {1}, {2}, {3}, {4})".format(trk_id, int(time.time()), self.camera, pos_x, pos_y)
				# print(info)
				self.cursor.execute(info)


	def insert_track(self, trk_id, track):
		# print("main type", type(track))
		# for key,ele in track.items():
		# 	print("dtype = ", key,type(ele))
		info = "SELECT track FROM TRACKS WHERE trk_id = {0}".format(trk_id)
		self.cursor.execute(info)
		data = self.cursor.fetchall()
		# print("Obtained tracks =",data)
		if len(data) == 0:
			info = "INSERT into TRACKS(trk_id, track) values ({0},'{1}')".format(trk_id,json.dumps(track))
			# print(info[:200])
			self.cursor.execute(info)

	def update_record(self, true_trk_id, false_trk_id):
		info = "UPDATE MOT SET person = {0} WHERE person = {1}".format(true_trk_id, false_trk_id)
		#print(info)
		self.cursor.execute(info)


	def update_track(self, true_trk_id, false_trk_id, track=None):
		if track:
			info = "UPDATE TRACKS SET track = '{0}' WHERE trk_id = {1}".format(json.dumps(track), false_trk_id)
		else:
			info = "UPDATE TRACKS SET trk_id = '{0}' WHERE trk_id = {1}".format(true_trk_id, false_trk_id)
		#print(info[:200])
		self.cursor.execute(info)

	def update_nextid(self, next_id):
		info = "UPDATE NEXTID SET next_id = {0} WHERE next_id = {1}".format(next_id + 1, next_id)
		# print(info)
		self.cursor.execute(info)

	
	def update_entry(self, total_ops):
		info = "UPDATE TOTAL SET entries = entries + {0} ".format(total_ops)
		# print(info)
		self.cursor.execute(info)

	def update_exit(self, total_ops):
		info = "UPDATE TOTAL SET exits = exits + {0} ".format(total_ops)
		# print(info)
		self.cursor.execute(info)

	def update_idtrack(self, origin_id, dupl_id, updateid=1, cam_id=None):
		# First to find duplicate
		if updateid == 1:
			info = "UPDATE TRACKS SET is_duplicate = {0} WHERE trk_id = {1}".format(origin_id, dupl_id)
			#print(info)
			self.cursor.execute(info)
		elif updateid != 1:
			info = "SELECT * FROM DUPL_COUNT WHERE dupl_id = {0}".format(dupl_id)
			self.cursor.execute(info)
			data = self.cursor.fetchall()
			# print("Pre_insert duplicates={} ,length={}".format(data,len(data)))

		info = "INSERT IGNORE INTO DUPL_COUNT SET cam_id = {0}, dupl_id = {1}, origin_id = {2}".format(cam_id, dupl_id, origin_id)
		#print(info)
		self.cursor.execute(info)

		info = "SELECT * FROM DUPL_COUNT WHERE dupl_id = {0}".format(dupl_id)
		self.cursor.execute(info)
		data = self.cursor.fetchall()
		# print("Length duplicates={}, {}".format(data,len(data)))
		if len(data) >= self.num_cams:
			info = "DELETE from TRACKS WHERE trk_id = {0}".format(dupl_id)
			#print(info)
			self.cursor.execute(info)

			info = "DELETE from DUPL_COUNT WHERE dupl_id = {0}".format(dupl_id)
			#print(info)
			self.cursor.execute(info)

			# info = "SELECT * FROM DUPL_COUNT WHERE dupl_id = {0}".format(dupl_id)
			# self.cursor.execute(info)
			# data = self.cursor.fetchall()
			# print("Length duplicates(POST DELETION)=",len(data))

	def delete_record(self, trk_id):
		info = "DELETE from MOT WHERE person = '{0}'".format(trk_id)
		#print(info)
		self.cursor.execute(info)

	def delete_track(self, trk_id):
		info = "DELETE from TRACKS WHERE trk_id = {0}".format(trk_id)
		#print(info)
		self.cursor.execute(info)

	def get_tracks(self):
		info = "SELECT track, is_duplicate FROM TRACKS"
		self.cursor.execute(info)
		data = self.cursor.fetchall()
		if len(data) > 0:
			tracks = []
			for byte_track in data:
				track = json.loads(byte_track[0])
				track["is_duplicate"] = byte_track[1]
				tracks.append(track)
				# print("Byte_track", byte_track, type(byte_track))
			return tracks

	def get_nextid(self):
		info = "SELECT next_id FROM NEXTID"
		self.cursor.execute(info)
		data = self.cursor.fetchall()
		if len(data) > 0:
			# print("Obtained next_id =",data)
			return data[0][0]

	def get_entry(self):
		info = "SELECT entries FROM TOTAL"
		self.cursor.execute(info)
		data = self.cursor.fetchall()
		if len(data) > 0:
			# print("Obtained data =",data)
			return data[0][0]

	def get_exit(self):
		info = "SELECT exits FROM TOTAL"
		self.cursor.execute(info)
		data = self.cursor.fetchall()
		if len(data) > 0:
			# print("Obtained data =",data)
			return data[0][0]

	def isExistTrkID(self, db_tracks, trk_id):
		try:
			return db_tracks[trk_id] != None
		except KeyError:
			return False



		