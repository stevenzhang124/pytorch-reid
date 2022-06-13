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

	def __init__(self, camera, database):
		self.database = database
		self.cursor = self.database.cursor()
		self.camera = camera
		# self.init()


	def init(self):
		self.cursor.execute("DROP TABLE IF EXISTS MOT")
		self.cursor.execute("DROP TABLE IF EXISTS TRACKS")
		self.cursor.execute("DROP TABLE IF EXISTS NEXTID")

		print("Deleted tables successfully")

		self.cursor.execute('''CREATE TABLE IF NOT EXISTS MOT
		(
		id int(11) unsigned NOT NULL AUTO_INCREMENT,
		person varchar(40) NOT NULL,
		ctime bigint(11) NOT NULL,
		camera int(11) NOT NULL,
		pos_x int(11) NOT NULL,
		pos_y int(11) NOT NULL,
		PRIMARY KEY (id)
		) ENGINE=InnoDB AUTO_INCREMENT=0 DEFAULT CHARSET=utf8;''')

		self.cursor.execute('''CREATE TABLE IF NOT EXISTS TRACKS
		(
		id int(11) unsigned NOT NULL AUTO_INCREMENT,
		trk_id bigint(11) NOT NULL,
		track LONGBLOB NOT NULL,
		is_duplicate int(1) DEFAULT 1 NOT NULL,
		PRIMARY KEY (id)
		) ENGINE=InnoDB AUTO_INCREMENT=0 DEFAULT CHARSET=utf8;''')

		self.cursor.execute('''CREATE TABLE IF NOT EXISTS NEXTID
		(
		id int(11) unsigned NOT NULL AUTO_INCREMENT,
		next_id int(11) NOT NULL,
		PRIMARY KEY (id)
		) ENGINE=InnoDB AUTO_INCREMENT=0 DEFAULT CHARSET=utf8;''')

		print("Created Tables successfully")
		self.database.commit()


		# Initialize Next_ID
		self.cursor.execute("INSERT into NEXTID(next_id) values ({0})".format(1))


	def insert_record(self, trk_id, pos_x, pos_y):
		info = "INSERT into MOT(person, ctime, camera, pos_x, pos_y) values ('{0}',{1},{2},{3},{4})".format(trk_id, int(time.time()), self.camera, pos_x, pos_y)
		print(info)
		self.cursor.execute(info)

	def insert_track(self, trk_id, track):
		# print("main type", type(track))
		# for key,ele in track.items():
		# 	print("dtype = ", key,type(ele))
		info = "SELECT track FROM TRACKS WHERE trk_id = '{0}'".format(trk_id)
		self.cursor.execute(info)
		data = self.cursor.fetchall()
		# print("Obtained tracks =",data)
		if len(data) == 0:
			info = "INSERT into TRACKS(trk_id, track) values ({0},'{1}')".format(trk_id,json.dumps(track))
			print(info[:200])
			self.cursor.execute(info)

	def update_record(self, true_trk_id, false_trk_id):
		info = "UPDATE MOT SET person = '{0}' WHERE person = '{1}'".format(true_trk_id, false_trk_id)
		print(info)
		self.cursor.execute(info)

	def update_track(self, true_trk_id, false_trk_id):
		info = "UPDATE TRACKS SET trk_id = '{0}' WHERE trk_id = '{1}'".format(true_trk_id, false_trk_id)
		print(info)
		self.cursor.execute(info)

	def update_nextid(self, next_id):
		info = "UPDATE NEXTID SET next_id = {0} WHERE next_id = {1}".format(next_id + 1, next_id)
		print(info)
		self.cursor.execute(info)

	def delete_record(self, trk_id):
		info = "DELETE from MOT WHERE person = '{0}'".format(trk_id)
		print(info)
		self.cursor.execute(info)

	def delete_track(self, trk_id, camera_num, dupl_bool=0):
		if dupl_bool != 0:
			info = "DELETE from TRACKS WHERE trk_id = '{0}'".format(trk_id)
			print(info)
			self.cursor.execute(info)
			return
		info = "UPDATE TRACKS SET is_duplicate = {0} WHERE trk_id = '{1}'".format(camera_num, trk_id)
		self.cursor.execute(info)
		info = "SELECT is_duplicate FROM TRACKS WHERE trk_id = '{0}'".format(trk_id)
		self.cursor.execute(info)
		data = self.cursor.fetchall()
		# print("Obtained tracks =",data)
		print("is_duplicate=",data[0])
		if data[0] == 2:
			info = "DELETE from TRACKS WHERE trk_id = '{0}'".format(trk_id)
			print(info)
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
		# print("Obtained tracks =",info)
		self.cursor.execute(info)
		data = self.cursor.fetchall()
		if len(data) > 0:
			return data[0][0]

	def isExistTrkID(self, db_tracks, trk_id):
		try:
			return db_tracks[trk_id] != None
		except KeyError:
			return False



		