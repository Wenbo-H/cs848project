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
DROP AGGREGATE IF EXISTS crf_agg(integer[], integer[], integer[], double precision[]) CASCADE;
DROP FUNCTION IF EXISTS crf_transit(double precision[], integer[], integer[], integer[], double precision[]) CASCADE;
DROP FUNCTION IF EXISTS crf_final(double precision[]) CASCADE;
DROP FUNCTION IF EXISTS crf_pre(double precision[], double precision[]) CASCADE;

CREATE FUNCTION crf_transit(double precision[], integer[], integer[], integer[], double precision[])
RETURNS double precision[]
AS 'crf-agg', 'grad'
LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION crf_final(double precision[])
RETURNS double precision[]
AS 'crf-agg', 'final'
LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION crf_pre(double precision[], double precision[])
RETURNS double precision[]
AS 'crf-agg', 'pre'
LANGUAGE C IMMUTABLE STRICT;

CREATE AGGREGATE crf_agg(integer[], integer[], integer[], double precision[]) (
	INITCOND = '{0}',
	STYPE = double precision[],
	PREFUNC = crf_pre,
	FINALFUNC = crf_final,
	SFUNC = crf_transit);

DROP FUNCTION IF EXISTS crf_loss(double precision[], integer[], integer[], integer[]) CASCADE;
CREATE FUNCTION crf_loss(double precision[], integer[], integer[], integer[])
RETURNS double precision
AS 'crf-agg', 'loss'
LANGUAGE C IMMUTABLE STRICT;

DROP FUNCTION IF EXISTS crf_pred(double precision[], integer[], integer[]) CASCADE;
CREATE FUNCTION crf_pred(double precision[], integer[], integer[])
RETURNS integer[]
AS 'crf-agg', 'pred'
LANGUAGE C IMMUTABLE STRICT;

DROP FUNCTION IF EXISTS crf_serialize(crf_model) CASCADE;
CREATE FUNCTION crf_serialize(crf_model)
RETURNS double precision[]
AS 'crf-agg', 'init'
LANGUAGE C IMMUTABLE STRICT;

DROP FUNCTION IF EXISTS crf_agg_iteration(data_table text, model_id integer) CASCADE;
CREATE FUNCTION crf_agg_iteration(data_table text, model_id integer)
RETURNS double precision AS $$
DECLARE
	weight_vector double precision[];
	loss double precision;
BEGIN
	-- grad
	EXECUTE 'SELECT crf_agg(uobs, bobs, labels, 
						    (SELECT crf_serialize(crf_model.*) 
							 FROM crf_model 
							 WHERE mid = ' || model_id || ')) '
			|| 'FROM ' || quote_ident(data_table)
		INTO weight_vector;
	-- update
	UPDATE crf_model SET w = weight_vector WHERE mid = model_id;
	UPDATE crf_model SET stepsize = (
			SELECT stepsize * decay FROM crf_model WHERE mid = model_id)
		WHERE mid = model_id;
	-- loss
	EXECUTE 'SELECT sum(crf_loss((SELECT crf_serialize(crf_model.*) 
								  FROM crf_model 
								  WHERE mid = ' || model_id || '),
						         uobs, bobs, labels)) '
			|| 'FROM ' || quote_ident(data_table)
		INTO loss;
	RETURN loss;
END;
$$ LANGUAGE plpgsql VOLATILE;

DROP FUNCTION IF EXISTS crf_train_agg(data_table text, model_id integer, iteration integer) CASCADE;
CREATE FUNCTION crf_train_agg(data_table text, model_id integer, iteration integer)
RETURNS VOID AS $$
DECLARE
	loss double precision;
BEGIN
	FOR i IN 1..iteration LOOP
		SELECT crf_agg_iteration(data_table, model_id) INTO loss;
		RAISE NOTICE '#iter: %, loss value: %', i, loss;
	END LOOP;
END;
$$ LANGUAGE plpgsql VOLATILE;

DROP FUNCTION IF EXISTS crf_eval(data_table text, model_id integer) CASCADE;
CREATE FUNCTION crf_eval(data_table text, model_id integer)
RETURNS double precision AS $$
DECLARE
	loss double precision;
BEGIN
	-- loss
	EXECUTE 'SELECT sum(crf_loss((SELECT crf_serialize(crf_model.*) 
								  FROM crf_model 
								  WHERE mid = ' || model_id || '),
						         uobs, bobs, labels)) '
			|| 'FROM ' || quote_ident(data_table)
		INTO loss;
	RETURN loss;
END;
$$ LANGUAGE plpgsql VOLATILE;

--------------------------------------------------------------------------
-- for shared memory version
--------------------------------------------------------------------------
DROP FUNCTION IF EXISTS crf_shmem_push(crf_model) CASCADE;
CREATE FUNCTION crf_shmem_push(crf_model)
RETURNS VOID
AS 'crf-shmem', 'init'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS crf_init(integer) CASCADE;
CREATE FUNCTION crf_init(model_id integer)
RETURNS VOID AS $$
DECLARE
	c integer;
BEGIN
	SELECT count(*) from crf_model WHERE mid = model_id INTO c;
	IF c < 1 THEN
		RAISE EXCEPTION 'No model with mid = % exists', model_id;
	ELSE
		PERFORM crf_shmem_push(crf_model.*) FROM crf_model WHERE mid = model_id; 
	END IF;
END;
$$ LANGUAGE plpgsql VOLATILE;

DROP FUNCTION IF EXISTS crf_clear(integer) CASCADE;
CREATE FUNCTION crf_clear(model_id integer)
RETURNS VOID AS $$
BEGIN
	PERFORM crf_shmem_pop(model_id);
END;
$$ LANGUAGE plpgsql VOLATILE;

DROP FUNCTION IF EXISTS crf_grad(integer, integer[], integer[], integer[]) CASCADE;
CREATE FUNCTION crf_grad(integer, integer[], integer[], integer[])
RETURNS VOID
AS 'crf-shmem', 'grad'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS crf_loss(integer, integer[], integer[], integer[]) CASCADE;
CREATE FUNCTION crf_loss(integer, integer[], integer[], integer[])
RETURNS double precision
AS 'crf-shmem', 'loss'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS crf_pred(integer, integer[], integer[]) CASCADE;
CREATE FUNCTION crf_pred(integer, integer[], integer[])
RETURNS integer[]
AS 'crf-shmem', 'pred'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS crf_shmem_pop(integer) CASCADE;
CREATE FUNCTION crf_shmem_pop(integer)
RETURNS double precision []
AS 'crf-shmem', 'final'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS crf_shmem_step(integer) CASCADE;
CREATE FUNCTION crf_shmem_step(integer)
RETURNS VOID
AS 'crf-shmem', 'pre'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS crf_shmem_iteration(data_table text, model_id integer) CASCADE;
CREATE FUNCTION crf_shmem_iteration(data_table text, model_id integer)
RETURNS double precision AS $$
DECLARE
	loss double precision;
BEGIN
	-- grad
	EXECUTE 'SELECT count(crf_grad(' || model_id || ', uobs, bobs, labels)) '
			|| 'FROM ' || quote_ident(data_table);
	-- update
	PERFORM crf_shmem_step(model_id);
	UPDATE crf_model SET stepsize = (
			SELECT stepsize * decay FROM crf_model WHERE mid = model_id)
		WHERE mid = model_id;
	-- loss
	EXECUTE 'SELECT sum(crf_loss(' || model_id || ', uobs, bobs, labels)) '
			|| 'FROM ' || quote_ident(data_table)
		INTO loss;
	RETURN loss;
END;
$$ LANGUAGE plpgsql VOLATILE;

DROP FUNCTION IF EXISTS crf_train_shmem(data_table text, model_id integer, iteration integer) CASCADE;
CREATE FUNCTION crf_train_shmem(data_table text, model_id integer, iteration integer)
RETURNS VOID AS $$
DECLARE
	loss double precision;
BEGIN
	PERFORM crf_shmem_push(crf_model.*) FROM crf_model WHERE mid = model_id;
	FOR i IN 1..iteration LOOP
		SELECT crf_shmem_iteration(data_table, model_id) INTO loss;
		RAISE NOTICE '#iter: %, loss value: %', i, loss;
	END LOOP;
	UPDATE crf_model SET w = (SELECT crf_shmem_pop(model_id)) WHERE mid = model_id;
END;
$$ LANGUAGE plpgsql VOLATILE;

--------------------------------------------------------------------------
-- wrappers
--------------------------------------------------------------------------
DROP FUNCTION IF EXISTS crf(text, integer, integer, integer, integer, integer, integer,
   	double precision, double precision, double precision, boolean, boolean) CASCADE;
CREATE FUNCTION crf(
	data_table text,
	model_id integer,
	ndims integer,
	nlabels integer,
	nulines integer,
	nblines integer,
	iteration integer /* default 20 */,
	mu double precision /* default 1e-1 */,
	stepsize double precision /* default 5e-2 */,
	decay double precision /* default 1 */,
	is_shmem boolean /* default 'false' */,
	is_shuffle boolean /* default 'true' */)
RETURNS VOID AS $$
DECLARE
	ntuples integer;
	tmp_table text;
	initw double precision[] := '{0}';
BEGIN
	-- query for dimension info and initialize the model table 
	EXECUTE 'SELECT count(*) FROM ' || data_table
		INTO ntuples;	
	SELECT alloc_float8_array(ndims) INTO initw;
	DELETE FROM crf_model WHERE mid = model_id;
	INSERT INTO crf_model VALUES (model_id, nlabels, ntuples, ndims, nulines, nblines,
	   	mu, stepsize, decay, initw); 
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
		PERFORM crf_train_shmem(tmp_table, model_id, iteration);
	ELSE
		PERFORM crf_train_agg(tmp_table, model_id, iteration);
	END IF;
END;
$$ LANGUAGE plpgsql VOLATILE;

DROP FUNCTION IF EXISTS crf(text, integer, integer, integer, integer, integer) CASCADE;
CREATE FUNCTION crf(
	data_table text,
	model_id integer,
	ndims integer,
	nlabels integer,
	nulines integer,
	nblines integer)
RETURNS VOID AS $$
	SELECT crf($1, $2, $3, $4, $5, $6, 20, 1e-1, 5e-2, 1, 'f', 't');
$$ LANGUAGE sql VOLATILE

