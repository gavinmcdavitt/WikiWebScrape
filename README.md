# WikiWebScrape
This use beautiful soup4 to scrape the internet of your favorite wikipedia articles.  I created this to help get data for a RAG model.

Run python script with a starting URL, it will then place the max amount of pages into a 'wikipedia_data.json' file. This file is what will be used in accordance with our RAG model.
 You will have a second file named: 'remaining_link.json' that will be able to put all of the future links you will want to use later. This application does work as a stack, so becareful it can take a while before some articles can be entered.
 

Script will grab wikipedia title, first section that is a summary of the page, and the URL. 


I used tom brady as training data to see if I could get a NLP model that would take some context and attempt to answer those questions. It was decent Knew how many super bowls he won but not how many he lost. So still working to get bot better, which might have a cost factor.
 Progress: We have a bunch of data that we need to connect our RAG machine. Then we should put an an analysis of all of the data, pick the article that is the best suited if over some percentage of correctness. Use the url to get the entire page and use that as context to answer a person's question. Boom you have an AI bot!!!!
