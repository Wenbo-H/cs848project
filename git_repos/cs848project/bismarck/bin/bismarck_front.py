"""
Copyright 2012 Xixuan (Aaron) Feng and Arun Kumar and Christopher Re

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
import imp
import psycopg2
from cStringIO import StringIO
import random

VERBOSE = False
SHUFFLE_PREFIX = '__bismarck_shuffled_'
MODELS = (
		'dense_logit', 'sparse_logit',
		'dense_svm', 'sparse_svm',
		'factor',
		'crf',
		)
PARAMS = {
		# required
		'model' : None,
		'model_id' : None,
		'data_table' : None,
		'feature_cols' : None,
		'label_col' : None,
		# except LMF
		'ndims' : None,
		# only for LMF
		'nrows' : None,
		'ncols' : None,
		'maxrank' : None,
		# only for CRF
		'nulines' : None,
		'nblines' : None,
		'nlabels' : None,
		# with default
		'num_iters' : 20,
		'initrange' : 0.01,
		'stepsize' : 0.1,
		'decay' : 1,
		'mu' : 1e-2,
		'B' : 2,
		'is_shmem' : False,
		'is_shuffle' : True,
		# optional
		'tolerance' : None,
		'output_file' : None,
		}

class DBInterface(object) :
	def __init__(self) :
		# connect DB using default connect string
		self.conn = psycopg2.connect('')

	def __del__(self) :
		self.conn.commit()
		self.conn.close()

	def execute(self, query) :
		cursor = self.conn.cursor()
		if VERBOSE :
			print 'executing: ', query
		cursor.execute(query)
		cursor.close()

	def execute_and_fetch(self, query) :
		cursor = self.conn.cursor()
		if VERBOSE :
			print 'executing: ', query
		cursor.execute(query)
		ret = cursor.fetchall()
		cursor.close()
		return ret

	def get_ntuples(self, data_table) :
		return self.execute_and_fetch(
				'SELECT count(*) FROM %s' % data_table)[0][0]

	def insert_model(self, model_table, model_id, w, **kwargs) :
		cursor = self.conn.cursor()
		# delete
		cursor.execute('DELETE FROM %s WHERE mid = %d' % 
				(model_table, model_id))
		# copy
		columns = ['mid', 'w']
		values = str(model_id) + '\t' + '{' + ','.join(map(str,w)) + '}'
		for k, v in kwargs.items() :
			columns.append(k)
			values += '\t' + str(v)
		cursor.copy_from(StringIO(values), model_table, columns=columns)
		cursor.close()

DB = DBInterface()

class Model(object) :
	def __init__(self) :
		self.data_table = PARAMS['data_table']
		self.feature_cols = PARAMS['feature_cols']
		self.label_col = PARAMS['label_col']
		self.model_id = PARAMS['model_id']
		self.ntuples = DB.get_ntuples(self.data_table)
		self.num_iters = PARAMS['num_iters']
		self.stepsize = PARAMS['stepsize']
		self.decay = PARAMS['decay']
		self.is_shmem = PARAMS['is_shmem']
		self.is_shuffle = PARAMS['is_shuffle']
		self.tolerance = PARAMS['tolerance']
		self.output_file = PARAMS['output_file']

	def prep(self) :
		self.insert_model_tuple()
		if self.is_shuffle :
			tmp_table = SHUFFLE_PREFIX + self.data_table + \
					'_' + str(self.model_id)
			DB.execute("""
				DROP TABLE IF EXISTS {0} CASCADE;
				CREATE TABLE {0} AS 
				SELECT {1}, {2} FROM {3} ORDER BY random();
				""".format(tmp_table, self.feature_cols, 
						self.label_col,	self.data_table))
			self.data_table = tmp_table
			print 'A shuffled table %s is created for training' % tmp_table
		if self.is_shmem :
			self.shmem_push()

	def iteration(self) :
		if self.is_shmem :
			self.shmem_grad()
			return self.shmem_loss()
		else :
			self.agg_grad()
			return self.agg_loss()

	def final(self) :
		if self.is_shmem :
			self.shmem_pop()
		if self.output_file is not None :
			self.output()

	def shmem_push(self) :
		DB.execute('SELECT {0}_shmem_push({1}.*) FROM {1} WHERE mid = {2}'
				.format(self.model, self.model_table, self.model_id))

	def shmem_pop(self) :
		DB.execute("""
			UPDATE {1} SET w = (SELECT {0}_shmem_pop({2})) 
			WHERE mid = {2}
			""".format(self.model, self.model_table, self.model_id))

	def shmem_grad(self) :
		DB.execute('SELECT count({0}_grad({1}, {2}, {3})) FROM {4}'
				.format(self.model, self.model_id, self.feature_cols, 
					self.label_col, self.data_table))
		DB.execute('SELECT {0}_shmem_step({1})'
				.format(self.model, self.model_id))

	def shmem_loss(self) :
		return DB.execute_and_fetch("""
			SELECT {0}({1}_loss({2}, {3}, {4})) FROM {5}
			""".format(self.agg, self.model, self.model_id, 
					self.feature_cols, self.label_col, self.data_table))[0][0]
	
	def agg_grad(self) :
		DB.execute("""
			UPDATE {0} SET w = (
				SELECT {1}_agg(
					{2}, {3},
					(SELECT {1}_serialize({0}.*) FROM {0} WHERE mid = {4})
					)
				FROM {5}
				)
			WHERE mid = {4}
			""".format(self.model_table, self.model, self.feature_cols, 
					self.label_col, self.model_id, self.data_table))
		DB.execute("""
			UPDATE {0} SET stepsize = 
					(SELECT stepsize * decay FROM {0} WHERE mid = {1})
			WHERE mid = {1}
			""".format(self.model_table, self.model_id))

	def agg_loss(self) :
		return DB.execute_and_fetch("""
			SELECT {0}({1}_loss(
				(SELECT {1}_serialize({2}.*) FROM {2} WHERE mid = {3}),
				{4}, {5}
				))
			FROM {6}
			""".format(self.agg, self.model, self.model_table, self.model_id,
					self.feature_cols, self.label_col, self.data_table))[0][0]
	
	def output(self) :
		self.w = DB.execute_and_fetch('SELECT w FROM {0} WHERE mid = {1}'
				.format(self.model_table, self.model_id))[0][0]
		fout = open(self.output_file, 'w')
		print >> fout, '\t'.join([str(_) for _ in self.w])
		fout.close()

class LinearModel(Model) :
	def __init__(self) :
		super(LinearModel, self).__init__()
		self.ndims = PARAMS['ndims']
		self.w = [0.0 for _ in range(self.ndims)]
		self.mu = PARAMS['mu']
		self.model_table = 'linear_model'
		self.agg = 'sum'

	def insert_model_tuple(self) :
		DB.insert_model(self.model_table, self.model_id, self.w,
				ntuples=self.ntuples, ndims=self.ndims, mu=self.mu,
				stepsize=self.stepsize, decay=self.decay)

class dense_logit(LinearModel) :
	def __init__(self) :
		super(dense_logit, self).__init__()
		self.model = 'dense_logit'
	
class sparse_logit(LinearModel) :
	def __init__(self) :
		super(sparse_logit, self).__init__()
		self.model = 'sparse_logit'
	
class dense_svm(LinearModel) :
	def __init__(self) :
		super(dense_svm, self).__init__()
		self.model = 'dense_svm'
	
class sparse_svm(LinearModel) :
	def __init__(self) :
		super(sparse_svm, self).__init__()
		self.model = 'sparse_svm'
	
class factor(Model) :
	def __init__(self) :
		super(factor, self).__init__()
		self.model = 'factor'
		self.nrows = PARAMS['nrows']
		self.ncols = PARAMS['ncols']
		self.maxrank = PARAMS['maxrank']
		self.ndims = (self.nrows + self.ncols) * self.maxrank
		self.initrange = PARAMS['initrange']
		self.w = [random.gauss(0, 1) * self.initrange for _ in range(self.ndims)]
		self.B = PARAMS['B']
		self.model_table = 'factor_model'
		self.agg = 'rmse'

	def insert_model_tuple(self) :
		DB.insert_model(self.model_table, self.model_id, self.w,
				ntuples=self.ntuples, ndims=self.ndims, B=self.B,
				stepsize=self.stepsize, decay=self.decay, nrows=self.nrows,
				ncols=self.ncols, maxrank=self.maxrank)

class crf(Model) :
	def __init__(self) :
		super(crf, self).__init__()
		self.model = 'crf'
		self.nulines = PARAMS['nulines']
		self.nblines = PARAMS['nblines']
		self.nlabels = PARAMS['nlabels']
		self.ndims = PARAMS['ndims']
		self.w = [0.0 for _ in range(self.ndims)]
		self.mu = PARAMS['mu']
		self.model_table = 'crf_model'
		self.agg = 'sum'

	def insert_model_tuple(self) :
		DB.insert_model(self.model_table, self.model_id, self.w,
				ntuples=self.ntuples, ndims=self.ndims, mu=self.mu,
				stepsize=self.stepsize, decay=self.decay, nulines=self.nulines,
				nblines=self.nblines, nlabels=self.nlabels)

def main() :
	# parameters from arguments
	spec = imp.load_source('bismarck.spec', sys.argv[1])
	for k in PARAMS :
		if k in spec.__dict__ :
			PARAMS[k] = spec.__dict__[k]
	
	# verbosing
	global VERBOSE
	try :
		VERBOSE = spec.verbose
	except AttributeError :
		VERBOSE = False
	
	# build the object for the specified model class
	model = None
	if spec.model in MODELS :
		model = globals()[spec.model]()
	else :
		print >> sys.stderr, 'model', spec.model, 'is not available'
		sys.exit(2)
	if VERBOSE :
		print 'attributes of model:'
		print [(k, v) for k, v in model.__dict__.items() if k != 'w']
	
	# main control block
	model.prep()
	previous_loss = 0.0
	for i in range(model.num_iters) :
		current_loss = model.iteration()
		# info
		improvement = None
		if i > 0 and previous_loss != 0.0:
			improvement = (previous_loss - current_loss) / previous_loss
		print 'iteration', i + 1, '\tloss:', current_loss,\
				'\timprovement: ', improvement
		# check tolerance
		if model.tolerance is not None and model.tolerance > improvement :
			break
		previous_loss = current_loss
	model.final()

if __name__ == '__main__' :
	if len(sys.argv) != 2 :
		print >> sys.stderr, 'Usage: python bismarck_front.py [spec_file]'
	else :
		random.seed()
		main()

