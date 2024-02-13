import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureCourseWarningSignals(BaseEstimator, TransformerMixin):
    """
    Generate features that capture the challenges a student faces in meeting deadlines for their coursework
    Args: term_class - dataframe, grades for courses for each student
    """
    
    def __init__(self, term_class:pd.DataFrame) -> None:    
        self.term_class = term_class
    
    def fit(self, X:pd.DataFrame, y:pd.Series=None) -> 'CustomTransformer':
        return self
    
    def grade_number_categories(self, grades_df, grade_col, n):
        """
        Groups number of grades into categoties: '1', '2', ...,'n and more'
        Creates dummy variables for these categories
    
        Args: grade_col - string, name of a column with grade statistics
        n - number of categories
        """
        grades_df = grades_df.copy()
    
        # Name for a new feature column
        col_name = "Grade_" + grade_col + "_num_of_courses"
        grades_df[col_name] = '0'
    
        # Group all values into "1" and "2 and more" categories
        for i in range(1, n):
            grades_df[col_name] = np.where(grades_df[grade_col]==i, str(i), grades_df[col_name])
    
        grades_df[col_name] = np.where(grades_df[grade_col]>=n, str(n)+' and more', grades_df[col_name])

        # Dummy variables
        for i in range(1, n):
            if i==1:
                col_name = "Grade_" + grade_col + "_for_" + str(i) + "_course"
            else:
                col_name = "Grade_" + grade_col + "_for_" + str(i) + "_courses"
            grades_df[col_name] = np.where(grades_df[grade_col]==i, 1, 0)
        
        col_name = "Grade_" + grade_col + "_for_" + str(n) + "_and_more_courses"
        grades_df[col_name] = np.where(grades_df[grade_col]>=n, 1, 0)
    
        return grades_df
    
    
    def transform(self, X:pd.DataFrame)  -> pd.DataFrame:
       
        transformed_X = X.copy()
        
        # Create dataframe with courses grades for each student in the sample
        grades_df = transformed_X[['STDNT_ID', 'TERM_CD']].merge(self.term_class,
                                                                 on=['STDNT_ID', 'TERM_CD'], how='left')
        
        grades_df=grades_df.groupby(['STDNT_ID', 'CRSE_GRD_INPUT_CD'])[['TERM_CD']].count()
        grades_df.reset_index(inplace=True)
        
        grades_df=grades_df.pivot(index=['STDNT_ID'], columns='CRSE_GRD_INPUT_CD', values='TERM_CD')
        
        
        # Not all students may have data on their grades for individual courses.
        # Create a flag that take a value of 1 to indicate that a student has information 
        # about individual course grades
        grades_list = list(grades_df.columns)
        try:
            grades_list.remove(' ')
        except BaseException:
            pass
        
        try:
            grades_list.remove('##')
        except BaseException:
            pass
        
        grades_df['CRSE_GRD_AVAILABLE'] = grades_df[grades_list].sum(axis=1)
        grades_df['CRSE_GRD_AVAILABLE'] = np.where(grades_df['CRSE_GRD_AVAILABLE']>=1,1,0)


        # I - Incomplete
        # Group all values into "1" and "2 and more" categories and create dummy variables
        grades_df = self.grade_number_categories(grades_df, 'I', 2)

        # ILE - "An "I" grade not finished by the incomplete deadline or an approved extended deadline lapses to "ILE""
        grades_df = self.grade_number_categories(grades_df, 'ILE', 2)

        # Grades starting with "I" (a student had an I grade in the past and finished the course)
        II = ['IA', 'IA-', 'IB+', 'IB', 'IB-', 'IC+', 'IC', 'IC-', 'ID+', 'ID',
              'ID-', 'IE', 'IF', 'ICR', 'INC', 'IP', 'IS']  
        grades_df['II'] = grades_df[II].sum(axis=1)
        # Group all values into "1" and "2 and more" 
        grades_df = self.grade_number_categories(grades_df, 'II', 2)

        # All Incomplete grades, including a grade 'NG' that has a similar meaning:
        # The NG is recorded when a student has been registered into a class after the web grade rosters 
        # have been sent to the instructor. The NG will convert to an ED* if unresolved after the first four weeks 
        # of the next fall or winter registration
        if 'NG' in list(grades_df.columns):
            grades_df['Overall_I'] = grades_df[['I', 'ILE', 'II', 'NG']].sum(axis=1)
        else:
            grades_df['Overall_I'] = grades_df[['I', 'ILE', 'II']].sum(axis=1)
        # Group all values into "1" and "2 and more" categories
        #grades_df = self.grade_number_categories(grades_df, 'Overall_I', 2)
        grades_df = self.grade_number_categories(grades_df, 'Overall_I', 1)

        # W - Official Withdrawal
        # Group all values into "1", "2", "3" and "4 and more" categories
        grades_df = self.grade_number_categories(grades_df, 'W', 3)

        # Y - "In these specially approved cases only, an instructor can report a Y grade at the end of the first-term 
        # course to indicate work in progress
        grades_df = self.grade_number_categories(grades_df, 'Y', 1)

        # NR - "The instructor should report an NR if a student stops attending before the end of the term, 
        # but has not dropped the class or requested an Incomplete."
        grades_df = self.grade_number_categories(grades_df, 'NR', 1)

        # Merge course warning signals with the main dataframe with features
        transformed_X = transformed_X.merge(grades_df, on=['STDNT_ID'], how='left')
        
        # Factor "No course grades at all"
        transformed_X['No_grades_at_all'] = np.where(transformed_X['CRSE_GRD_AVAILABLE'].isnull(), 1, 0)
        
        # We take into account the absence of grade data in the 'No_grades_at_all' column, 
        # so the other columns should not contain 'NaN' values
        grade_cols = [col for col in transformed_X.columns if col.startswith('Grade_')]
        for grade_col in grade_cols:
            transformed_X[grade_col] = np.where(transformed_X[grade_col].isnull(),
                                                0, transformed_X[grade_col])
        return transformed_X