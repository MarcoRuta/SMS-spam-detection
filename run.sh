python3 utility/data_analysis.py
echo -e "\e[1;31m Blacklist approach \e[0m"
python3 blacklist.py
echo -e "\e[1;31m LSH approach \e[0m"
python3 locality_sensitive_hashing.py
echo -e "\e[1;31m CountVectorizer approach \e[0m"
python3 count_vectorizer.py

