/*
 * demographics.h
 *
 *  Created on: 23 Mar 2020
 *      Author: hinchr
 */

#ifndef DEMOGRAPHICS_H_
#define DEMOGRAPHICS_H_

/************************************************************************/
/****************************** Structures  *****************************/
/************************************************************************/

struct demographic_household_table{
	long n_households;
	long n_total;
	long *idx;
	int *age_group;
	long *house_no;
};

struct demographic_occupation_network_table {
    long n_networks;
    long n_total;
    long *network_no; // network assigned to each person
    long *network_size;
    long *mean_interactions;
    double *lockdown_occupation_multipliers;
    long *network_ids;
    char **network_names;
};

/************************************************************************/
/******************************  Functions  *****************************/
/************************************************************************/

void generate_household_distribution( model* );
void assign_household_distribution( model*, demographic_household_table* );
void set_up_household_distribution( model* );
void set_up_allocate_work_places( model* );
void set_up_allocate_default_work_places( model* );
void set_up_allocate_custom_work_places( model* );
void build_household_network_from_directroy( network*, directory* );
void add_reference_household( double *, long , int **);
	
#endif /* DEMOGRAPHICS_H_ */
