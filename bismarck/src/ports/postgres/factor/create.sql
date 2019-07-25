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
DROP AGGREGATE IF EXISTS factor_agg(integer, integer, double precision, double precision[]) CASCADE;
DROP FUNCTION IF EXISTS factor_transit(double precision[], integer, integer, double precision, double precision[]) CASCADE;
DROP FUNCTION IF EXISTS factor_final(double precision[]) CASCADE;
DROP FUNCTION IF EXISTS factor_pre(double precision[], double precision[]) CASCADE;

CREATE FUNCTION factor_transit(double precision[], integer, integer, double precision, double precision[])
RETURNS double precision[]
AS 'factor-agg', 'grad'
LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION factor_final(double precision[])
RETURNS double precision[]
AS 'factor-agg', 'final'
LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION factor_pre(double precision[], double precision[])
RETURNS double precision[]
AS 'factor-agg', 'pre'
LANGUAGE C IMMUTABLE STRICT;

CREATE AGGREGATE factor_agg(integer, integer, double precision, double precision[]) (
	INITCOND = '{0}',
	STYPE = double precision[],
	PREFUNC = factor_pre,
	FINALFUNC = factor_final,
	SFUNC = factor_transit);

DROP FUNCTION IF EXISTS factor_loss(double precision[], integer, integer, double precision) CASCADE;
CREATE FUNCTION factor_loss(double precision[], integer, integer, double precision)
RETURNS double precision
AS 'factor-agg', 'loss'
LANGUAGE C IMMUTABLE STRICT;

DROP FUNCTION IF EXISTS factor_pred(double precision[], integer, integer) CASCADE;
CREATE FUNCTION factor_pred(double precision[], integer, integer)
RETURNS double precision
AS 'factor-agg', 'loss'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS factor_serialize(factor_model) CASCADE;
CREATE FUNCTION factor_serialize(factor_model)
RETURNS double precision[]
AS 'factor-agg', 'init'
LANGUAGE C IMMUTABLE STRICT;

DROP FUNCTION IF EXISTS factor_agg_iteration(data_table text, model_id integer) CASCADE;
CREATE FUNCTION factor_agg_iteration(data_table text, model_id integer)
RETURNS double precision AS $$
DECLARE
	weight_vector double precision[];
	loss double precision;
BEGIN
	-- grad
	EXECUTE 'SELECT factor_agg(row, col, rating, 
						    (SELECT factor_serialize(factor_model.*) 
							 FROM factor_model 
							 WHERE mid = ' || model_id || ')) '
			|| 'FROM ' || quote_ident(data_table)
		INTO weight_vector;
	-- update
	UPDATE factor_model SET w = weight_vector WHERE mid = model_id;
	UPDATE factor_model SET stepsize = (
			SELECT stepsize * decay FROM factor_model WHERE mid = model_id)
		WHERE mid = model_id;
	-- loss
	EXECUTE 'SELECT rmse(factor_loss((SELECT factor_serialize(factor_model.*) 
								  FROM factor_model 
								  WHERE mid = ' || model_id || '),
						         row, col, rating)) '
			|| 'FROM ' || quote_ident(data_table)
		INTO loss;
	RETURN loss;
END;
$$ LANGUAGE plpgsql VOLATILE;

DROP FUNCTION IF EXISTS factor_train_agg(data_table text, model_id integer, iteration integer) CASCADE;
CREATE FUNCTION factor_train_agg(data_table text, model_id integer, iteration integer)
RETURNS VOID AS $$
DECLARE
	loss double precision;
BEGIN
	FOR i IN 1..iteration LOOP
		SELECT factor_agg_iteration(data_table, model_id) INTO loss;
		RAISE NOTICE '#iter: %, RMSE: %', i, loss;
	END LOOP;
END;
$$ LANGUAGE plpgsql VOLATILE;

DROP FUNCTION IF EXISTS factor_eval(data_table text, model_id integer) CASCADE;
CREATE FUNCTION factor_eval(data_table text, model_id integer)
RETURNS double precision AS $$
DECLARE
	loss double precision;
BEGIN
	-- loss
	EXECUTE 'SELECT rmse(factor_loss((SELECT factor_serialize(crf_model.*) 
								  FROM factor_model 
								  WHERE mid = ' || model_id || '),
						         row, col, rating)) '
			|| 'FROM ' || quote_ident(data_table)
		INTO loss;
	RETURN loss;
END;
$$ LANGUAGE plpgsql VOLATILE;

--------------------------------------------------------------------------
-- for shared memory version
--------------------------------------------------------------------------
DROP FUNCTION IF EXISTS factor_shmem_push(factor_model) CASCADE;
CREATE FUNCTION factor_shmem_push(factor_model)
RETURNS VOID
AS 'factor-shmem', 'init'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS factor_init(integer) CASCADE;
CREATE FUNCTION factor_init(model_id integer)
RETURNS VOID AS $$
DECLARE
	c integer;
BEGIN
	SELECT count(*) FROM factor_model WHERE mid = model_id INTO c;
	IF c < 1 THEN
		RAISE EXCEPTION 'No model with mid = % exists', model_id;
	ELSE
		PERFORM factor_shmem_push(factor_model.*) FROM factor_model WHERE mid = $1;
	END IF;
END;
$$ LANGUAGE plpgsql VOLATILE;

DROP FUNCTION IF EXISTS factor_clear(integer) CASCADE;
CREATE FUNCTION factor_clear(model_id integer)
RETURNS VOID AS $$
BEGIN
	PERFORM factor_shmem_pop(model_id);
END;
$$ LANGUAGE plpgsql VOLATILE;

DROP FUNCTION IF EXISTS factor_grad(integer, integer, integer, double precision) CASCADE;
CREATE FUNCTION factor_grad(integer, integer, integer, double precision)
RETURNS VOID
AS 'factor-shmem', 'grad'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS factor_loss(integer, integer, integer, double precision) CASCADE;
CREATE FUNCTION factor_loss(integer, integer, integer, double precision)
RETURNS double precision
AS 'factor-shmem', 'loss'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS factor_pred(integer, integer, integer) CASCADE;
CREATE FUNCTION factor_pred(integer, integer, integer)
RETURNS double precision
AS 'factor-shmem', 'loss'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS factor_shmem_pop(integer) CASCADE;
CREATE FUNCTION factor_shmem_pop(integer)
RETURNS double precision []
AS 'factor-shmem', 'final'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS factor_shmem_step(integer) CASCADE;
CREATE FUNCTION factor_shmem_step(integer)
RETURNS VOID
AS 'factor-shmem', 'pre'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS factor_shmem_iteration(data_table text, model_id integer) CASCADE;
CREATE FUNCTION factor_shmem_iteration(data_table text, model_id integer)
RETURNS double precision AS $$
DECLARE
	loss double precision;
BEGIN
	-- grad
	EXECUTE 'SELECT count(factor_grad(' || model_id || ', row, col, rating)) '
			|| 'FROM ' || quote_ident(data_table);
	-- update
	PERFORM factor_shmem_step(model_id);
	UPDATE factor_model SET stepsize = (
			SELECT stepsize * decay FROM factor_model WHERE mid = model_id)
		WHERE mid = model_id;
	-- loss
	EXECUTE 'SELECT rmse(factor_loss(' || model_id || ', row, col, rating)) '
			|| 'FROM ' || quote_ident(data_table)
		INTO loss;
	RETURN loss;
END;
$$ LANGUAGE plpgsql VOLATILE;

DROP FUNCTION IF EXISTS factor_train_shmem(data_table text, model_id integer, iteration integer) CASCADE;
CREATE FUNCTION factor_train_shmem(data_table text, model_id integer, iteration integer)
RETURNS VOID AS $$
DECLARE
	loss double precision;
BEGIN
	PERFORM factor_shmem_push(factor_model.*) FROM factor_model WHERE mid = model_id;
	FOR i IN 1..iteration LOOP
		SELECT factor_shmem_iteration(data_table, model_id) INTO loss;
		RAISE NOTICE '#iter: %, RMSE: %', i, loss;
	END LOOP;
	UPDATE factor_model SET w = (SELECT factor_shmem_pop(model_id)) WHERE mid = model_id;
END;
$$ LANGUAGE plpgsql VOLATILE;

--------------------------------------------------------------------------
-- wrappers
--------------------------------------------------------------------------
DROP FUNCTION IF EXISTS factor(text, integer, integer, integer, integer, integer, double precision,
   	double precision, double precision, double precision, boolean, boolean) CASCADE;
CREATE FUNCTION factor(
	data_table text,
	model_id integer,
	nrows integer,
	ncols integer,
	maxrank integer,
	iteration integer /* default 20 */,
	b double precision /* default 2 */,
	initrange double precision /* default 1e-2 */,
	stepsize double precision /* default 1e-2 */,
	decay double precision /* default 1 */,
	is_shmem boolean /* default 'false' */,
	is_shuffle boolean /* default 'true' */)
RETURNS VOID AS $$
DECLARE
	ndims integer;
	ntuples integer;
--	mean double precision;
--	temp double precision;
	tmp_table text;
	initw double precision[] := '{0}';
BEGIN
	-- query for ntuples and initialize the model table 
	ndims := (nrows + ncols) * maxrank;
	EXECUTE 'SELECT count(*) FROM ' || data_table
		INTO ntuples;	
--	EXECUTE 'SELECT sqrt(avg(rating) / ' || maxrank || ') FROM ' || data_table
--		INTO mean;
--	RAISE NOTICE 'mean: %', mean;
	SELECT alloc_float8_array_random(ndims, initrange) INTO initw;
--	FOR i IN 1..ndims LOOP
--		SELECT (1.0 - random()) * mean * 2 INTO temp;
--		initw[i] := temp;
--	END LOOP;
	DELETE FROM factor_model WHERE mid = model_id;
	INSERT INTO factor_model VALUES (model_id, nrows, ncols, maxrank, ndims, 
		ntuples, b, stepsize, decay, initw); 
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
		PERFORM factor_train_shmem(tmp_table, model_id, iteration);
	ELSE
		PERFORM factor_train_agg(tmp_table, model_id, iteration);
	END IF;
END;
$$ LANGUAGE plpgsql VOLATILE;

DROP FUNCTION IF EXISTS factor(text, integer, integer, integer, integer) CASCADE;
CREATE FUNCTION factor(
	data_table text,
	model_id integer,
	nrows integer,
	ncols integer,
	maxrank integer)
RETURNS VOID AS $$
	SELECT factor($1, $2, $3, $4, $5, 20, 2, 1e-2, 1e-2, 1, 'f', 't');
$$ LANGUAGE sql VOLATILE

