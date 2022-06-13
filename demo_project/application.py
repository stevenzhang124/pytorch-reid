from flask import Flask, render_template, request, jsonify
import mysql.connector
from mysql.connector import errorcode
import json
# import pymysql
# from flaskext.mysql import MySQL
import cv

from datetime import datetime

application = Flask(__name__, static_url_path='/static')


# application = Flask(__name__)


@application.route('/')
def index():
    if request.host[-4:] == "5000":
        # return render_template('index_vis.html')
        return render_template('index_4cam.html')
    else:
        return render_template('index_access.html')


# def db_connection():
#     mysql = MySQL()
#
#     application.config['MYSQL_DATABASE_HOST'] = "localhost"
#     application.config['MYSQL_DATABASE_PORT'] = 3306
#     application.config['MYSQL_DATABASE_USER'] = "root"
#     application.config['MYSQL_DATABASE_PASSWORD'] = "edgedemo"  # voucherapp
#     application.config['MYSQL_DATABASE_DB'] = "edgedemo"
#
#
#     mysql.init_app(app=application)
#
#     return mysql.get_db().cursor()

def db_connection():
    try:
      mydb = mysql.connector.connect(host='192.168.1.103',  # 192.168.1.100
                                       user='cqy',
                                       port='3306',
                                       database='edgedemo',
                                       passwd='123456',
                                       autocommit=True)
    except mysql.connector.Error as err:
      if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("Something is wrong with your user name or password")
      elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("Database does not exist")
      else:
        print(err)
    else:
        return mydb


def executeSQL(sql_statement, arg=None):
    connection = db_connection()
    cursor = connection.cursor()
    print("Opened database successfully")
    try:
        print("The statement and arguments are",sql_statement,arg)
        if arg:
            cursor.execute(sql_statement, (arg,))
        else:
            cursor.execute(sql_statement)
        data = cursor.fetchall()
        # print("Data of executeSQL() ", data)
        if len(data) > 0:
            print("Found records in Database", data)
            return jsonify(data), 200
        else:
            response = {'message': 'No data found'}
            return jsonify(response), 200
    except mysql.connector.IntegrityError:
        print("Failed to obtain values from database")
        response = {'message': 'No data found'}
        return jsonify(response), 404

    finally:
        cursor.close()


@application.route('/query', methods=['POST', 'GET'])
def query():
    if request.method == "POST":
        print("QUERY'S Request is",request.form)
        personID = request.form['person']
        # print("Person is", personID)
        sql_persons = """SELECT person, ctime, camera, pos_x, pos_y, gender, age, hair, luggage, attire FROM MOT WHERE PERSON = %s"""

        executed_stmt = json.loads(executeSQL(sql_persons, arg=personID)[0].data.decode('utf-8'))
        print("Pre_response", executed_stmt)

        transform = cv.transform(h, executed_stmt, 3, 4)

        print("Post_response", transform)

        # transform[0] contains only first element here
        response = jsonify(transform)

        return response


@application.route('/queryPersons', methods=['GET'])
def queryPersons():
    sql_persons = """SELECT person FROM MOT """

    return executeSQL(sql_persons)


@application.route('/getHomography', methods=['GET'])
def getHomography():
    global h
    h = cv.init()
    return str(len(h)), 200

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-p', '--port', required=True, type=int, help='port to listen on')
    args = parser.parse_args()
    application.run(port=args.port, debug=True)
