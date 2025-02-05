import pandas as pd

def transform_data(df):
    # Step 1: Rename columns
    df.rename(columns={
        'CNTSCHID': 'schoolID',
        'CNTSTUID': 'studentID',
        'ST004D01T': 'Gender',
        'SC001Q01TA': 'municipality_size',
        'SC037Q02TA': 'external_evaluation',
        'SC187Q04WA': 'math_grouping',
        'SC011Q01TA': 'degree_of_competition',
        'SC180Q01JA': 'additional_math_classes',
        'SC218Q01JA': 'track_student_attendance'
    }, inplace=True)

    # Step 2: Convert `schoolID` and `studentID` to categorical
    df['schoolID'] = df['schoolID'].astype('category')
    df['studentID'] = df['studentID'].astype('category')

    # Step 3: Filter out columns with more than 15% missing data
    threshold = 0.15
    missing_values = df.isnull().mean()
    cols_to_drop = missing_values[missing_values > threshold].index
    df = df.drop(cols_to_drop, axis=1)
    
    # Step 4: Replace track categories
    track_categories = {
        'Italy : Vocational Institute, Art Institute (Industry and Craft Workers)': 'Vocational',
        'Italy : Artistic, Classical, Linguistic, Music & performing arts, Scientific (applied science), Social (socio-economic)': 'Academic',
        'Italy : Technical Institute': 'Technical',
        'Italy : Vocational training; Vocational Schools of Bolzano and Trento Provinces': 'Vocational Schools (Bolzano & Trento)',
        'Italy : Lower secondary education': 'Lower Secondary Education'
    }
    df['PROGN'] = df['PROGN'].replace(track_categories)

    # Step 5: Remove specific categories from 'PROGN'
    categories_to_remove = ['Vocational Schools (Bolzano & Trento)', 'Lower Secondary Education']
    df = df[~df['PROGN'].isin(categories_to_remove)]
    
    # Step 6: Simplify language
    def simplify_language(lang):
        if lang == 'Italian' or pd.isna(lang):
            return lang
        elif lang == 'Missing':
            return pd.NA
        else:
            return 'Other'
    
    df['LANGN'] = df['LANGN'].apply(simplify_language)
    
    # Step 7: Simplify immigration status
    def simplify_immig_status(immig):
        return 'Native student' if immig == 'Native student' else 'Immig student'
    
    df['IMMIG'] = df['IMMIG'].apply(simplify_immig_status)
    
    # Step 8: Simplify municipality size
    df.loc[df['municipality_size'].isin(['A village, hamlet or rural area (fewer than 3 000 people)']), 'municipality_size'] = 'A town (15 000 to about 100 000 people)'
    df.loc[df['municipality_size'].isin(['A large city (1 000 000 to about 10 000 000 people)']), 'municipality_size'] = 'A city (100 000 to about 1 000 000 people)'
    
    # Step 9: Simplify class size
    df.loc[df['CLSIZE'].isin(['More than 50 students', '46-50 students']), 'CLSIZE'] = '21-25 students'
    
    # making SCHAUTO and EDUSHORT ordinal variables (see the original distributions in the notebook)
    bins = [-float('inf'), -1, 0, 1, float('inf')] 
    labels = [1, 2, 3, 4] 
    df['SCHAUTO_ORDINAL'] = pd.cut(df['SCHAUTO'], bins=bins, labels=labels, include_lowest=True)

    bins = [-float('inf'), -0.5, 0.5, float('inf')]  
    labels = ['Low', 'Middle', 'High']  
    df['EDUSHORT_binned'] = pd.cut(df['EDUSHORT'], bins=bins, labels=labels)
    
    df.drop(columns=['SCHAUTO', 'EDUSHORT', 'HISCED'], inplace= True)
    
    return df






