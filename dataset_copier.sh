# /bin/bash

NB_ROOT=$1

nbs=(
  "lextoumbourou/feedback3-eda-hf-custom-trainer-sift"
  "paultimothymooney/kaggle-survey-2022-all-results"
  "dataranch/supermarket-sales-prediction-xgboost-fastai"
  "kkhandekar/environmental-vs-ai-startups-india-eda"
  "ampiiere/animal-crossing-villager-popularity-analysis"
  "aieducation/what-course-are-you-going-to-take"
  "saisandeepjallepalli/adidas-retail-eda-data-visualization"
  "joshuaswords/netflix-data-visualization"
  "spscientist/student-performance-in-exams"
  "ibtesama/getting-started-with-a-movie-recommendation-system"
  "nickwan/creating-player-stats-using-tracking-data"
  "erikbruin/nlp-on-student-writing-eda"
  "madhurpant/beautiful-kaggle-2022-analysis"
  "pmarcelino/comprehensive-data-exploration-with-python"
  "gksriharsha/eda-speedtests"
  "mpwolke/just-you-wait-rishi-sunak"
  "sanket7994/imdb-dataset-eda-project"
  "roopacalistus/retail-supermarket-store-analysis"
  "sandhyakrishnan02/indian-startup-growth-analysis"
  "roopacalistus/exploratory-data-analysis-retail-supermarket"
  "sandhyakrishnan02/indian-startup-growth-analysis"
  "roopacalistus/exploratory-data-analysis-retail-supermarket"
  "brianmendieta/data-cleaning-plus-eda"
  "deffro/eda-is-fun"
  "artgor/eda-and-models"
  "kanncaa1/dataiteam-titanic-eda"
  "shivavashishtha/zomato-eda-tutorial"
  "khoongweihao/covid-19-novel-coronavirus-eda-forecasting-cases"
  "carlmcbrideellis/simple-eda-of-kaggle-grandmasters-scheduled"
  "willkoehrsen/start-here-a-gentle-introduction"
  "vanguarde/h-m-eda-first-look"
  "yuliagm/talkingdata-eda-plus-time-patterns"
  "arimishabirin/globalsalary-simple-eda"
  "beratozmen/clash-of-clans-exploratory-data-analysis"
  "itzsanju/eda-airline-dataset"
  "jyotsananegi/melbourne-housing-snapshot-eda"
  "madseth/customer-shopping-trends-dataset-eda"
  "mikedelong/python-eda-with-kdes"
  "nicoleashley/iit-admission-eda"
  "roopahegde/cryptocurrency-price-correlation"
  "saniaks/melbourne-house-price-eda"
  "sunnybiswas/eda-on-airline-dataset"
  "akshaypetkar/supermarket-sales-analysis"
  "qnqfbqfqo/electric-vehicle-landscape-in-washington-state"
  "xokent/cyber-security-attack-eda"
  "josecode1/billionaires-statistics-2023"
  "mathewvondersaar/analysis-of-student-performance",
  "viviktpharale/house-price-prediction-eda-linear-ridge-lasso",
  "jagangupta/stop-the-s-toxic-comments-eda",
  "kimtaehun/simple-preprocessing-for-time-series-prediction",
  "vatsalmavani/music-recommendation-system-using-spotify-dataset",
  "tanulsingh077/twitter-sentiment-extaction-analysis-eda-and-model"
)

# Assumption: All datasets are moved to the Datasets_Extracted director in the project root folder
for nb in ${nbs[@]}; do
  echo ${nb}
  cp -r Datasets_Extracted/${nb}/input ${NB_ROOT}/${nb}/
done
