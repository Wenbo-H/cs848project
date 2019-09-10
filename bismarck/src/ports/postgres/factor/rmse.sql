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

DROP FUNCTION IF EXISTS _float8_rmse(double precision[]) CASCADE;
CREATE FUNCTION _float8_rmse(double precision[])
RETURNS FLOAT8 AS $$
	SELECT SQRT($1[2] / $1[1]);
$$ LANGUAGE SQL;

DROP AGGREGATE IF EXISTS rmse(double precision);
CREATE AGGREGATE rmse (double precision) (
    sfunc = float8_accum,
    stype = float8[],
    prefunc = float8_amalg,
    finalfunc = _float8_rmse,
    initcond = '{0,0,0}');

