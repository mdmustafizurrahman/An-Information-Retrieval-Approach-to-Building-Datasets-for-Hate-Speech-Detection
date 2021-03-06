# There are 3 types of csv file here
  1. pilot_all.csv contains all tweets (4,668 tweets) collected and annotated in pilot phase
  2. main.csv and main_all.csv contains (4,999 tweets) collected and annotated in the second stage of the annotation 
  3. main_subset.csv contains a subset of tweets (4,725 tweets) after discarding tweets 

# CSV file format
1. tweetID<br/>
2. final_label1<br/>
3. hate_auto_label1<br/>
4. expliciteHateType1<br/>
5. hatecategory1<br/>
6. highlighted_terms1<br/>
7. highlighted_terms_label1<br/>
8. ImplicithateType1<br/>
9. Implicittaret1<br/>
10. rationale1<br/>
11. final_label2<br/>
12. hate_auto_label2<br/>
13. expliciteHateType2<br/>
14. hatecategory2<br/>
15. highlighted_terms2<br/>
16. highlighted_terms_label2<br/>
17. ImplicithateType2<br/>
18. Implicittaret2<br/>
19. rationale2<br/>
20. final_label3<br/>
21. hate_auto_label3<br/>
22. expliciteHateType3<br/>
23. hatecategory3<br/>
24. highlighted_terms3<br/>
25. highlighted_terms_label3<br/>
26. ImplicithateType3<br/>
27. Implicittaret3<br/>
28. rationale3<br/>
29. majority_label<br/>
30. majority_label_auto<br/>

Each row contains 30 colums, where there are 9 columns for each annotators. These 9 columns are:
1. final_label<annotators#id> -- final binary hate label (hate or non-hate) provided directly by the annotator
2. hate_auto_label<annotators#id> -- final binary hate label (hate or non-hate) inferred on the basis of the annotator's answer to annotation sub-tasks in identifying 1) derogatory language or language inciting violence; and 2) target demographic group
3. expliciteHateType<annotators#id> -- if the hate type (incite violence, deragatory langauge or None) is explicit, this column captures that
4. hatecategory<annotators#id> -- targeted groups for the hate (race, gender, etc) is captured by this column  
5. highlighted_terms<annotators#id> -- terms highlighted by the annotators related to hate type and hate category 
6. highlighted_terms_label<annotators#id> -- labels of the terms highlighted by the annotators related to hate type and hate category  
7. ImplicithateType<annotators#id> -- if the hate type (incite violence, deragatory langauge or None) is implicit, this column captures that
8. Implicittaret<annotators#id> -- if the targeted group of hate is implicit, this column captures that
9. rationale<annotators#id> -- any justification provided by the annotators for their decisions 

where <annotators#id> in {1, 2, 3} indicates the first, second, or third annotator for the given tweet. Note that this does not refer to an unique identifier for the annotator across tweets.

## Self Consistency Check
For self consistencey check, we use these two fields 1) final_label<annotator#id>, and 2) hate_auto_label<annotator#id> for each annotator. 
 
