from pathlib import Path
import shutil
names_map = {
    'notebooks/sanket7994/imdb-dataset-eda-project/bench.ipynb': 'imdb',
    'notebooks/madhurpant/beautiful-kaggle-2022-analysis/bench.ipynb': 'mad-kaggle',
    'notebooks/sandhyakrishnan02/indian-startup-growth-analysis/bench.ipynb': 'startups',
    'notebooks/nickwan/creating-player-stats-using-tracking-data/bench.ipynb': 'player-stats',
    'notebooks/kkhandekar/environmental-vs-ai-startups-india-eda/bench.ipynb': 'india-startups',
    'notebooks/josecode1/bench.ipynb': 'billionaires',
    'notebooks/pmarcelino/comprehensive-data-exploration-with-python/bench.ipynb': 'data-exploration',
    'notebooks/roopacalistus/retail-supermarket-store-analysis/bench.ipynb': 'supermarket-analysis',
    'notebooks/gksriharsha/eda-speedtests/bench.ipynb': 'speedtests',
    'notebooks/saisandeepjallepalli/adidas-retail-eda-data-visualization/bench.ipynb': 'adidas',
    'notebooks/spscientist/student-performance-in-exams/bench.ipynb': 'exams',
    'notebooks/aieducation/what-course-are-you-going-to-take/bench.ipynb': 'course',
    'notebooks/dataranch/supermarket-sales-prediction-xgboost-fastai/bench.ipynb': 'supermarket-prediction',
    'notebooks/lextoumbourou/feedback3-eda-hf-custom-trainer-sift/bench.ipynb': 'feedback3',
    'notebooks/ampiiere/animal-crossing-villager-popularity-analysis/bench.ipynb': 'animal-crossing',
    'notebooks/mpwolke/just-you-wait-rishi-sunak/bench.ipynb': 'sunak',
    'notebooks/erikbruin/nlp-on-student-writing-eda/bench.ipynb': 'nlp-writing',
    'notebooks/ibtesama/getting-started-with-a-movie-recommendation-system/bench.ipynb': 'movie-rec',
    'notebooks/joshuaswords/netflix-data-visualization/bench.ipynb': 'netflix-viz'
}

def main():
    for name, replacement_stem in names_map.items():
        p = Path(name)
        if p.exists():
            parent_dir = p.parent
            shutil.copy2(p, parent_dir / f'{replacement_stem}.ipynb')

if __name__ == '__main__':
    main(   )