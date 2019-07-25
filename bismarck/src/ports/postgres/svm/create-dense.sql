/*
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
*/

--------------------------------------------------------------------------
-- for UDA version
--------------------------------------------------------------------------
DROP AGGREGATE IF EXISTS dense_svm_agg(double precision[], integer, double precision[]) CASCADE;
DROP FUNCTION IF EXISTS dense_svm_transit(double precision[], double precision[], integer, double precision[]) CASCADE;
DROP FUNCTION IF EXISTS dense_svm_final(double precision[]) CASCADE;
DROP FUNCTION IF EXISTS dense_svm_pre(double precision[], double precision[]) CASCADE;

CREATE FUNCTION dense_svm_transit(double precision[], double precision[], integer, double precision[])
RETURNS double precision[]
AS 'dense-svm-agg', 'grad'
LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION dense_svm_final(double precision[])
RETURNS double precision[]
AS 'dense-svm-agg', 'final'
LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION dense_svm_pre(double precision[], double precision[])
RETURNS double precision[]
AS 'dense-svm-agg', 'pre'
LANGUAGE C IMMUTABLE STRICT;

CREATE AGGREGATE dense_svm_agg(double precision[], integer, double precision[]) (
	INITCOND = '{0}',
	STYPE = double precision[],
	PREFUNC = dense_svm_pre,
	FINALFUNC = dense_svm_final,
	SFUNC = dense_svm_transit);

DROP FUNCTION IF EXISTS dense_svm_loss(double precision[], double precision[], integer) CASCADE;
CREATE FUNCTION dense_svm_loss(double precision[], double precision[], integer)
RETURNS double precision
AS 'dense-svm-agg', 'loss'
LANGUAGE C IMMUTABLE STRICT;

DROP FUNCTION IF EXISTS dense_svm_pred(double precision[], double precision[]) CASCADE;
CREATE FUNCTION dense_svm_pred(double precision[], double precision[])
RETURNS double precision
AS 'dense-svm-agg', 'pred'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS dense_svm_serialize(linear_model) CASCADE;
CREATE FUNCTION dense_svm_serialize(linear_model)
RETURNS double precision[]
AS 'dense-svm-agg', 'init'
LANGUAGE C IMMUTABLE STRICT;

DROP FUNCTION IF EXISTS dense_svm_agg_iteration(data_table text, model_id integer) CASCADE;
CREATE FUNCTION dense_svm_agg_iteration(data_table text, model_id integer)
RETURNS double precision AS $$
DECLARE
	weight_vector double precision[];
	loss double precision;
BEGIN
	-- grad
	EXECUTE 'SELECT dense_svm_agg(vec, labeli, 
						    (SELECT dense_svm_serialize(linear_model.*) 
							 FROM linear_model 
							 WHERE mid = ' || model_id || ')) '
			|| 'FROM ' || quote_ident(data_table)
		INTO weight_vector;
	-- update
	UPDATE linear_model SET w = weight_vector WHERE mid = model_id;
	UPDATE linear_model SET stepsize = (
			SELECT stepsize * decay FROM linear_model WHERE mid = model_id)
		WHERE mid = model_id;
	-- loss
	EXECUTE 'SELECT sum(dense_svm_loss((SELECT dense_svm_serialize(linear_model.*) 
								  FROM linear_model 
								  WHERE mid = ' || model_id || '),
						         vec, labeli)) '
			|| 'FROM ' || quote_ident(data_table)
		INTO loss;
	RETURN loss;
END;
$$ LANGUAGE plpgsql VOLATILE;

DROP FUNCTION IF EXISTS dense_svm_train_agg(data_table text, model_id integer, iteration integer) CASCADE;
CREATE FUNCTION dense_svm_train_agg(data_table text, model_id integer, iteration integer)
RETURNS VOID AS $$
DECLARE
	loss double precision;
BEGIN
	FOR i IN 1..iteration LOOP
		SELECT dense_svm_agg_iteration(data_table, model_id) INTO loss;
		RAISE NOTICE '#iter: %, loss value: %', i, loss;
	END LOOP;
END;
$$ LANGUAGE plpgsql VOLATILE;

DROP FUNCTION IF EXISTS dense_svm_eval(data_table text, model_id integer) CASCADE;
CREATE FUNCTION dense_svm_eval(data_table text, model_id integer)
RETURNS double precision AS $$
DECLARE
	loss double precision;
BEGIN
	-- loss
	EXECUTE 'SELECT sum(dense_svm_loss((SELECT dense_svm_serialize(linear_model.*) 
								  FROM linear_model 
								  WHERE mid = ' || model_id || '),
						         vec, labeli)) '
			|| 'FROM ' || quote_ident(data_table)
		INTO loss;
	RETURN loss;
END;
$$ LANGUAGE plpgsql VOLATILE;

--------------------------------------------------------------------------
-- for shared memory version
--------------------------------------------------------------------------
DROP FUNCTION IF EXISTS dense_svm_shmem_push(linear_model) CASCADE;
CREATE FUNCTION dense_svm_shmem_push(linear_model)
RETURNS VOID
AS 'dense-svm-shmem', 'init'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS dense_svm_init(integer) CASCADE;
CREATE FUNCTION dense_svm_init(model_id integer)
RETURNS VOID AS $$
DECLARE
	c integer;
BEGIN
	SELECT count(*) FROM linear_model WHERE mid = model_id INTO c;
	IF c < 1 THEN
		RAISE EXCEPTION 'No model with mid = % exists', model_id;
	ELSE
		PERFORM dense_svm_shmem_push(linear_model.*) FROM linear_model WHERE mid = model_id;
	END IF;
END;
$$ LANGUAGE plpgsql VOLATILE;

DROP FUNCTION IF EXISTS dense_svm_clear(integer) CASCADE;
CREATE FUNCTION dense_svm_clear(model_id integer)
RETURNS VOID AS $$
BEGIN
	PERFORM dense_svm_shmem_pop(model_id);
END;
$$ LANGUAGE plpgsql VOLATILE;

DROP FUNCTION IF EXISTS dense_svm_grad(integer, double precision[], integer) CASCADE;
CREATE FUNCTION dense_svm_grad(integer, double precision[], integer)
RETURNS VOID
AS 'dense-svm-shmem', 'grad'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS dense_svm_loss(integer, double precision[], integer) CASCADE;
CREATE FUNCTION dense_svm_loss(integer, double precision[], integer)
RETURNS double precision
AS 'dense-svm-shmem', 'loss'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS dense_svm_pred(integer, double precision[]) CASCADE;
CREATE FUNCTION dense_svm_pred(integer, double precision[])
RETURNS double precision
AS 'dense-svm-shmem', 'pred'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS dense_svm_shmem_pop(integer) CASCADE;
CREATE FUNCTION dense_svm_shmem_pop(integer)
RETURNS double precision []
AS 'dense-svm-shmem', 'final'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS dense_svm_shmem_step(integer) CASCADE;
CREATE FUNCTION dense_svm_shmem_step(integer)
RETURNS VOID
AS 'dense-svm-shmem', 'pre'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS dense_svm_shmem_iteration(data_table text, model_id integer) CASCADE;
CREATE FUNCTION dense_svm_shmem_iteration(data_table text, model_id integer)
RETURNS double precision AS $$
DECLARE
	loss double precision;
BEGIN
	-- grad
	EXECUTE 'SELECT count(dense_svm_grad(' || model_id || ', vec, labeli)) '
			|| 'FROM ' || quote_ident(data_table);
	-- update
	PERFORM dense_svm_shmem_step(model_id);
	UPDATE linear_model SET stepsize = (
			SELECT stepsize * decay FROM linear_model WHERE mid = model_id)
		WHERE mid = model_id;
	-- loss
	EXECUTE 'SELECT sum(dense_svm_loss(' || model_id || ', vec, labeli)) '
			|| 'FROM ' || quote_ident(data_table)
		INTO loss;
	RETURN loss;
END;
$$ LANGUAGE plpgsql VOLATILE;

DROP FUNCTION IF EXISTS dense_svm_train_shmem(data_table text, model_id integer, iteration integer) CASCADE;
CREATE FUNCTION dense_svm_train_shmem(data_table text, model_id integer, iteration integer)
RETURNS VOID AS $$
DECLARE
	loss double precision;
BEGIN
	PERFORM dense_svm_shmem_push(linear_model.*) FROM linear_model WHERE mid = model_id;
	FOR i IN 1..iteration LOOP
		SELECT dense_svm_shmem_iteration(data_table, model_id) INTO loss;
		RAISE NOTICE '#iter: %, loss value: %', i, loss;
	END LOOP;
	UPDATE linear_model SET w = (SELECT dense_svm_shmem_pop(model_id)) WHERE mid = model_id;
END;
$$ LANGUAGE plpgsql VOLATILE;

--------------------------------------------------------------------------
-- wrappers
--------------------------------------------------------------------------
DROP FUNCTION IF EXISTS dense_svm(text, integer, integer, integer, double precision,
	double precision, double precision, boolean, boolean) CASCADE;
CREATE FUNCTION dense_svm(
	data_table text,
	model_id integer,
	ndims integer,
	iteration integer /* default 20 */,
	mu double precision /* default 1e-2 */,
	stepsize double precision /* default 5e-5 */,
	decay double precision /* default 1 */,
	is_shmem boolean /* default 'false' */,
	is_shuffle boolean /* default 'true' */)
RETURNS VOID AS $$
DECLARE
	ntuples integer;
	tmp_table text;
	initw double precision[] := '{0}';
BEGIN
	-- query for ntuples and initialize the model table 
	EXECUTE 'SELECT count(*) FROM ' || data_table
		INTO ntuples;	
	SELECT alloc_float8_array(ndims) INTO initw;
	DELETE FROM linear_model WHERE mid = model_id;
	INSERT INTO linear_model VALUES (model_id, ndims, ntuples, mu, stepsize, decay, initw); 
	-- execute iterations
	IF is_shuffle THEN
		tmp_table := '__bismarck_shuffled_' || data_table || '_' || model_id;
		EXECUTE 'DROP TABLE IF EXISTS ' || tmp_table || ' CASCADE';
		EXECUTE 'CREATE TABLE ' || tmp_table || ' AS 
			SELECT * FROM ' || data_table || ' ORDER BY random()';
		RAISE NOTICE 'A shuffled table % is created for training', tmp_table;
	ELSE
		tmp_table := data_table;
	END IF;
	IF is_shmem THEN
		PERFORM dense_svm_train_shmem(tmp_table, model_id, iteration);
	ELSE
		PERFORM dense_svm_train_agg(tmp_table, model_id, iteration);
	END IF;
END;
$$ LANGUAGE plpgsql VOLATILE;

DROP FUNCTION IF EXISTS dense_svm(text, integer, integer) CASCADE;
CREATE FUNCTION dense_svm(
	data_table text,
	model_id integer,
	ndims integer)
RETURNS VOID AS $$
	SELECT dense_svm($1, $2, $3, 20, 1e-2, 5e-5, 1, 'f', 't');
$$ LANGUAGE sql VOLATILE

