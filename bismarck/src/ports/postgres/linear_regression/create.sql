
DROP AGGREGATE IF EXISTS linear_reg_agg(double precision[], integer, double precision[]) CASCADE;
DROP FUNCTION IF EXISTS linear_reg_transit(double precision[], double precision[], integer, double precision[]) CASCADE;
DROP FUNCTION IF EXISTS linear_reg_final(double precision[]) CASCADE;
DROP FUNCTION IF EXISTS linear_reg_pre(double precision[], double precision[]) CASCADE;


CREATE FUNCTION linear_reg_transit(double precision[], double precision[], integer, double precision[])
RETURNS double precision[]
AS 'linear-reg-agg', 'grad'
LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION linear_reg_final(double precision[])
RETURNS double precision[]
AS 'linear-reg-agg', 'final'
LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION linear_reg_pre(double precision[], double precision[])
RETURNS double precision[]
AS 'linear-reg-agg', 'pre'
LANGUAGE C IMMUTABLE STRICT;

CREATE AGGREGATE linear_reg_agg(double precision[], integer, double precision[]) (
	INITCOND = '{0}',
	STYPE = double precision[],
	PREFUNC = linear_reg_pre,
	FINALFUNC = linear_reg_final,
	SFUNC = linear_reg_transit);

DROP FUNCTION IF EXISTS linear_reg_loss(double precision[], double precision[], integer) CASCADE;
CREATE FUNCTION linear_reg_loss(double precision[], double precision[], integer)
RETURNS double precision
AS 'linear-reg-agg', 'loss'
LANGUAGE C IMMUTABLE STRICT;

DROP FUNCTION IF EXISTS linear_reg_pred(double precision[], double precision[]) CASCADE;
CREATE FUNCTION linear_reg_pred(double precision[], double precision[])
RETURNS double precision
AS 'linear-reg-agg', 'pred'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS linear_reg_serialize(linear_model) CASCADE;
CREATE FUNCTION linear_reg_serialize(linear_model)
RETURNS double precision[]
AS 'linear-reg-agg', 'init'
LANGUAGE C IMMUTABLE STRICT;


DROP FUNCTION IF EXISTS linear_reg_agg_iteration(data_table text, model_id integer) CASCADE;
CREATE FUNCTION linear_reg_agg_iteration(data_table text, model_id integer)
RETURNS double precision AS $$
DECLARE
	weight_vector double precision[];
	loss double precision;
BEGIN
    RAISE NOTICE 'start grad';
	-- grad
	EXECUTE 'SELECT linear_reg_agg(vec, price, 
						    (SELECT linear_reg_serialize(linear_model.*) 
							 FROM linear_model 
							 WHERE mid = ' || model_id || ')) '
			|| 'FROM ' || quote_ident(data_table)
		INTO weight_vector;
    RAISE NOTICE 'finish grad';
	-- update
	UPDATE linear_model SET w = weight_vector WHERE mid = model_id;
	UPDATE linear_model SET stepsize = (
			SELECT stepsize * decay FROM linear_model WHERE mid = model_id)
		WHERE mid = model_id;
	-- loss
	EXECUTE 'SELECT sum(linear_reg_loss((SELECT linear_reg_serialize(linear_model.*) 
								  FROM linear_model 
								  WHERE mid = ' || model_id || '),
						         vec, price)) '
			|| 'FROM ' || quote_ident(data_table)
		INTO loss;
	RETURN loss;
END;
$$ LANGUAGE plpgsql VOLATILE;

DROP FUNCTION IF EXISTS linear_reg_train_agg(data_table text, model_id integer, iteration integer) CASCADE;
CREATE FUNCTION linear_reg_train_agg(data_table text, model_id integer, iteration integer)
RETURNS VOID AS $$
DECLARE
	loss double precision;
BEGIN
	FOR i IN 1..iteration LOOP
        RAISE NOTICE 'start #iter: %, table: %, mid: %, iteration: %,', i, data_table, model_id, iteration;
		SELECT linear_reg_agg_iteration(data_table, model_id) INTO loss;
		RAISE NOTICE '#iter: %, loss value: %', i, loss;
	END LOOP;
END;
$$ LANGUAGE plpgsql VOLATILE;

DROP FUNCTION IF EXISTS linear_reg_eval(data_table text, model_id integer) CASCADE;
CREATE FUNCTION linear_reg_eval(data_table text, model_id integer)
RETURNS double precision AS $$
DECLARE
	loss double precision;
BEGIN
	-- loss
	EXECUTE 'SELECT sum(linear_reg_loss((SELECT linear_reg_serialize(linear_model.*) 
								  FROM linear_model 
								  WHERE mid = ' || model_id || '),
						         vec, price)) '
			|| 'FROM ' || quote_ident(data_table)
		INTO loss;
	RETURN loss;
END;
$$ LANGUAGE plpgsql VOLATILE;

--------------------------------------------------------------------------
-- for shared memory version
--------------------------------------------------------------------------
DROP FUNCTION IF EXISTS linear_reg_shmem_push(linear_model) CASCADE;
CREATE FUNCTION linear_reg_shmem_push(linear_model)
RETURNS VOID
AS 'linear-reg-shmem', 'init'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS linear_reg_init(integer) CASCADE;
CREATE FUNCTION linear_reg_init(model_id integer)
RETURNS VOID AS $$
DECLARE
	c integer;
BEGIN
	SELECT count(*) from linear_model WHERE mid = model_id INTO c;
	IF c < 1 THEN
		RAISE EXCEPTION 'No model with mid = % exists', model_id;
	ELSE
		PERFORM linear_reg_shmem_push(linear_model.*) FROM linear_model WHERE mid = model_id; 
	END IF; 
END;
$$ LANGUAGE plpgsql VOLATILE;
									
DROP FUNCTION IF EXISTS linear_reg_clear(integer) CASCADE;
CREATE FUNCTION linear_reg_clear(model_id integer)
RETURNS VOID AS $$
BEGIN
	PERFORM linear_reg_shmem_pop(model_id);
END;
$$ LANGUAGE plpgsql VOLATILE;

DROP FUNCTION IF EXISTS linear_reg_grad(integer, double precision[], integer) CASCADE;
CREATE FUNCTION linear_reg_grad(integer, double precision[], integer)
RETURNS VOID
AS 'linear-reg-shmem', 'grad'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS linear_reg_loss(integer, double precision[], integer) CASCADE;
CREATE FUNCTION linear_reg_loss(integer, double precision[], integer)
RETURNS double precision
AS 'linear-reg-shmem', 'loss'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS linear_reg_pred(integer, double precision[]) CASCADE;
CREATE FUNCTION linear_reg_pred(integer, double precision[])
RETURNS double precision
AS 'linear-reg-shmem', 'pred'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS linear_reg_shmem_pop(integer) CASCADE;
CREATE FUNCTION linear_reg_shmem_pop(integer)
RETURNS double precision []
AS 'linear-reg-shmem', 'final'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS linear_reg_shmem_step(integer) CASCADE;
CREATE FUNCTION linear_reg_shmem_step(integer)
RETURNS VOID
AS 'linear-reg-shmem', 'pre'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS linear_reg_shmem_iteration(data_table text, model_id integer) CASCADE;
CREATE FUNCTION linear_reg_shmem_iteration(data_table text, model_id integer)
RETURNS double precision AS $$
DECLARE
	loss double precision;
BEGIN
	-- grad
	EXECUTE 'SELECT count(linear_reg_grad(' || model_id || ', vec, price)) '
			|| 'FROM ' || quote_ident(data_table);
	-- update
	PERFORM linear_reg_shmem_step(model_id);
	UPDATE linear_model SET stepsize = (
			SELECT stepsize * decay FROM linear_model WHERE mid = model_id)
		WHERE mid = model_id;
	-- loss
	EXECUTE 'SELECT sum(linear_reg_loss(' || model_id || ', vec, price)) '
			|| 'FROM ' || quote_ident(data_table)
		INTO loss;
	RETURN loss;
END;
$$ LANGUAGE plpgsql VOLATILE;

DROP FUNCTION IF EXISTS linear_reg_train_shmem(data_table text, model_id integer, iteration integer) CASCADE;
CREATE FUNCTION linear_reg_train_shmem(data_table text, model_id integer, iteration integer)
RETURNS VOID AS $$
DECLARE
	loss double precision;
BEGIN
	PERFORM linear_reg_shmem_push(linear_model.*) FROM linear_model WHERE mid = model_id;
	FOR i IN 1..iteration LOOP
		SELECT linear_reg_shmem_iteration(data_table, model_id) INTO loss;
		RAISE NOTICE '#iter: %, loss value: %', i, loss;
	END LOOP;
	UPDATE linear_model SET w = (SELECT linear_reg_shmem_pop(model_id)) WHERE mid = model_id;
END;
$$ LANGUAGE plpgsql VOLATILE;

--------------------------------------------------------------------------
-- wrappers
--------------------------------------------------------------------------
DROP FUNCTION IF EXISTS linear_reg(text, integer, integer, integer, double precision,
	double precision, double precision, boolean, boolean) CASCADE;
CREATE FUNCTION linear_reg(
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
		PERFORM linear_reg_train_shmem(tmp_table, model_id, iteration);
	ELSE
        RAISE NOTICE 'start training linear regression';
		PERFORM linear_reg_train_agg(tmp_table, model_id, iteration);
	END IF;
END;
$$ LANGUAGE plpgsql VOLATILE;

DROP FUNCTION IF EXISTS linear_reg(text, integer, integer) CASCADE;
CREATE FUNCTION linear_reg(
	data_table text,
	model_id integer,
	ndims integer)
RETURNS VOID AS $$
	SELECT linear_reg($1, $2, $3, 20, 1e-2, 5e-5, 1, 'f', 't');
$$ LANGUAGE sql VOLATILE

