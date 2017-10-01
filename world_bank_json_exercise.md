
<h1>0. Introduction.</h1>

As an exercise in working with JSON files, we will work with a data set from the World Bank that provides information about how money lended to different countries is utilized.

For this assignment specifically, we'd like to answer the following:

<ol>
<li>Find the 10 countries with most projects.</li>
<li>Find the top 10 major project themes (using column 'mjtheme_namecode').</li>
<li>In 2. above you will notice that some entries have only the code and the name is missing. Create a dataframe with the missing names filled in.</li>
</ol>

To start, we can load the necessary Python packages and the JSON file provided to see what we have.


```python
import pandas as pd
import json
from pandas.io.json import json_normalize

# Load file as data frame and see what columns we have
world_bank_df = pd.read_json('data/world_bank_projects.json')
print('Data dimensions: ', world_bank_df.shape)
print(world_bank_df.columns)
print('Total number of projects: ', world_bank_df.id.nunique())
```

    Data dimensions:  (500, 50)
    Index(['_id', 'approvalfy', 'board_approval_month', 'boardapprovaldate',
           'borrower', 'closingdate', 'country_namecode', 'countrycode',
           'countryname', 'countryshortname', 'docty', 'envassesmentcategorycode',
           'grantamt', 'ibrdcommamt', 'id', 'idacommamt', 'impagency',
           'lendinginstr', 'lendinginstrtype', 'lendprojectcost',
           'majorsector_percent', 'mjsector_namecode', 'mjtheme',
           'mjtheme_namecode', 'mjthemecode', 'prodline', 'prodlinetext',
           'productlinetype', 'project_abstract', 'project_name', 'projectdocs',
           'projectfinancialtype', 'projectstatusdisplay', 'regionname', 'sector',
           'sector1', 'sector2', 'sector3', 'sector4', 'sector_namecode',
           'sectorcode', 'source', 'status', 'supplementprojectflg', 'theme1',
           'theme_namecode', 'themecode', 'totalamt', 'totalcommamt', 'url'],
          dtype='object')
    Total number of projects:  500


We find that we have a total of 500 different projects for analysis with each project having 50 different attributes to describe it.

<hr>
<h1>1. Find the 10 countries with the most projects.</h1>

Based on the columns that we have, the projects can be grouped according to the country names (<code>'countrynames'</code>). We can then count
the number of projects in each group and sort the results in descending order. The countries
with a greater number of projects will be towards the top of this list.



```python
proj_per_country = world_bank_df.groupby('countryname')['id'].size() # Group according to 'countryname'
sorted_countries = proj_per_country.sort_values(ascending=False)     # Keep the greatest values at the top
```

<p> Our goal is to find the ten countries with the most projects, but it's <b>important to check for 
any tied values</b>. Rather than take a slice of the first ten items from <code>'sorted_countries'</code>,
we can look at, say, the top fifteen items listed, just in case multiple countries share the same number of projects.</p>


```python
print(sorted_countries[:15]) # Check for ties by taking a slice with the first 15 values.
```

    countryname
    People's Republic of China         19
    Republic of Indonesia              19
    Socialist Republic of Vietnam      17
    Republic of India                  16
    Republic of Yemen                  13
    Nepal                              12
    People's Republic of Bangladesh    12
    Kingdom of Morocco                 12
    Africa                             11
    Republic of Mozambique             11
    Burkina Faso                        9
    Federative Republic of Brazil       9
    Islamic Republic of Pakistan        9
    United Republic of Tanzania         8
    Republic of Tajikistan              8
    Name: id, dtype: int64


The 10th and 11th elements in this ranked list, Republic of Mozambique and Burkina Faso, 
respectively, are not tied. Thus, our final answer for the 10 countries would just
be a slice of the first 10 elements, which is listed in the below table.


```python
print('The 10 countries with the most projects:')
print('----------------------------------------')
print(sorted_countries[:10]) # No ties, so just slice the first 10 in the sorted data.
```

    The 10 countries with the most projects:
    ----------------------------------------
    countryname
    People's Republic of China         19
    Republic of Indonesia              19
    Socialist Republic of Vietnam      17
    Republic of India                  16
    Republic of Yemen                  13
    Nepal                              12
    People's Republic of Bangladesh    12
    Kingdom of Morocco                 12
    Africa                             11
    Republic of Mozambique             11
    Name: id, dtype: int64


<hr>
<h1>2. Find the top 10 major project themes (use 'mjtheme_namecode').</h1>

First, we should look at the data structure for the <code>'mjtheme_namecode'</code> column.
It turns out that it is a list of dictionaries, and some projects listed are described by more than
one code. If we want to count the major project themes, these data should be expanded for a proper count.

A great way to do this is to use the <code>'json_normalize'</code> function to work with these data, and make
a table which we will call <code>'mjtable'</code>.


```python
# Load and create a table with mjtheme_namecode
wb_list = json.load(open('data/world_bank_projects.json'))
mjtable = json_normalize(wb_list, 'mjtheme_namecode') # Create table to expand the listed project codes
```

Each entry in the <code>'mjtheme_namecode'</code> column has <code>'code'</code> and corresponding <code>'name'</code>.
It seems reasonable to group these data by <code>'name'</code>, but we will find that values are missing and this will
be a group on its own. This is not desirable, because some categories will be underrepresented in the final tally.


```python
sorted_mjthemes_byname = mjtable.groupby('name').size().sort_values(ascending=False) # Groups based on name
print(sorted_mjthemes_byname[:15]) # Check the 15 highest values for ties
```

    name
    Environment and natural resources management    223
    Rural development                               202
    Human development                               197
    Public sector governance                        184
    Social protection and risk management           158
    Financial and private sector development        130
                                                    122
    Social dev/gender/inclusion                     119
    Trade and integration                            72
    Urban development                                47
    Economic management                              33
    Rule of law                                      12
    dtype: int64


There are 122 entries that are missing names. However, the values for 
the <code>'code'</code> column are all present. Creating groups with these data
instead will provide the correct values. We can then take a slice of the top 15
as we previously did to see if there are any tied categories.


```python
sorted_mjthemes_bycode = mjtable.groupby('code').size().sort_values(ascending=False) # Groups based on code
print(sorted_mjthemes_bycode[:15]) # Check the 15 highest values in case of ties
```

    code
    11    250
    10    216
    8     210
    2     199
    6     168
    4     146
    7     130
    5      77
    9      50
    1      38
    3      15
    dtype: int64


The counts are a correct representation of the data, and there are no ties.

It would be helpful if we convert these numeric codes into their descriptive
names, however. We can address this by creating a dictionary for the different themes,
and then look up the names for the top ten categories.


```python
# How many entries are we expecting for the dictionary?
print('Total number of themes: ', str(mjtable['code'].nunique()))
# Build the dictionary from the subset of data without missing values.

complete = mjtable[mjtable['name'] != '']   # Note: for this particular case, we're lucky, because
                                            # all possible themes will be represented here
    
mjtheme_dict = complete.set_index('code')['name'].to_dict() # Create dictionary
print('Total number of dictionary entries: ', str(len(mjtheme_dict)))

# Use the dictionary to convert 'code' back to their descriptive names
# Note: We keep the top 10, since we already saw there were no ties
print('----------------------------------------------------------')
mjthemes_top10 = sorted_mjthemes_bycode[:10].to_frame('counts')
mjthemes_top10['description'] = list(map(mjtheme_dict.get, mjthemes_top10.index))
print(mjthemes_top10)
```

    Total number of themes:  11
    Total number of dictionary entries:  11
    ----------------------------------------------------------
          counts                                   description
    code                                                      
    11       250  Environment and natural resources management
    10       216                             Rural development
    8        210                             Human development
    2        199                      Public sector governance
    6        168         Social protection and risk management
    4        146      Financial and private sector development
    7        130                   Social dev/gender/inclusion
    5         77                         Trade and integration
    9         50                             Urban development
    1         38                           Economic management


The ten themes with the greatest number of projects are listed in the above table.

<hr>
<h1>3. Create a dataframe with the missing names filled in.</b></h1>

It would be helpful to replace all the missing values with the appropriate theme names,
and we can use our dictionary to do this. Furthermore, after replacing the missing terms, we can 
save the data frame to a JSON file for future use.


```python
filled_namecodes = world_bank_df['mjtheme_namecode'] # Pick out the column we want

for data in filled_namecodes:
    length = len(data) # Iterate through the list of codes for the project
    for idx in range(0,length):
        # If the 'name' information is missing
        if (data[idx]['name'] == ''):
            # look it up in the dictionary and replace it
            data[idx]['name'] = mjtheme_dict[data[idx]['code']] 
            
# Save the DataFrame to a JSON file
world_bank_df.to_json('data/world_bank_projects_filled.json', orient='records') 
```

As a way to check, we can load the JSON file with completed names, and verify that all data are 
present in the <code>'mjtheme_namecodes'</code> column.


```python
# Load the completed JSON file
check_wb_list = json.load(open('data/world_bank_projects_filled.json'))
check_mjtable = json_normalize(check_wb_list, 'mjtheme_namecode')
# Check if there are any missing values
number_missing = len(check_mjtable[check_mjtable['name'] == ''])
print('There are ', str(number_missing), 'entries missing.')
```

    There are  0 entries missing.


Previously, we had 122 values missing, and now there are none in the updated JSON file. 

Note: we can also load this file as a data frame as we originally did, and continue to work with it as we wish.


```python
check_world_bank_df = pd.read_json('world_bank_projects_filled.json') # Load the new, completed data as a data frame
```
