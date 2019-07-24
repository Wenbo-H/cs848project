MODULES := crf logit svm factor linear array
PGPORTDIR := src/ports/postgres/
MODULEDIRS := $(addprefix $(PGPORTDIR),$(MODULES))
PGMODULES := $(MODULEDIRS:%=%-pg)
GPMODULES := $(MODULEDIRS:%=%-gp)

.PHONY: pg $(PGMODULES) gp $(GPMODULES)
	
pg: $(PGMODULES) 
gp: $(GPMODULES)

$(PGMODULES):
	make -C $(@:%-pg=%) pg

$(GPMODULES):
	make -C $(@:%-gp=%) gp

SQLFILES := $(foreach dir,$(MODULEDIRS),$(shell find $(dir) -name 'create*.sql')) 
MODELTABLES := $(foreach dir,$(MODULEDIRS),$(shell find $(dir) -name '*model.sql')) 
AGGREGATES := $(foreach dir,$(MODULEDIRS),$(shell find $(dir) -name 'rmse.sql')) 
ARRAY_FUNCS := $(foreach dir,$(MODULEDIRS),$(shell find $(dir) -name 'array.sql')) 

.PHONY: install-pg install-gp $(SQLFILES) $(MODELTABLES) $(AGGREGATES) $(ARRAY_FUNCS)

install-pg: $(SQLFILES)
install-gp: $(SQLFILES)

$(SQLFILES): $(MODELTABLES) $(AGGREGATES)
	psql -f $@

$(AGGREGATES): $(ARRAY_FUNCS)
	psql -f $@

$(MODELTABLES):
	psql -f $@

$(ARRAY_FUNCS):
	psql -f $@

