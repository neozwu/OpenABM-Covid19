%module model_utils

%inline %{
int utils_n_current( model *model, int type ) {
    return model->event_lists[type].n_current;
}

int utils_n_total( model *model, int type ) {
    return model->event_lists[type].n_total;
}

int utils_n_total_age( model *model, int type, int age ) {
    return model->event_lists[type].n_total_by_age[age];
}

int utils_n_total_occupation( model *model, int type, int network ) {
    return model->event_lists[type].n_total_by_occupation[network];
}

long utils_occupation_size( model *model, int network ) {
    return model->occupation_network[network]->n_vertices;
}

char* utils_occupation_name( model *model, int network ) {
    return model->occupation_network[network]->name;
}

int utils_n_daily( model *model, int type, int day ) {
    return model->event_lists[type].n_daily_current[day];
}
%}


%extend model{
    ~model() {
        destroy_model($self);
    }
}
