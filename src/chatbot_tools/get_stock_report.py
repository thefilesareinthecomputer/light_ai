@staticmethod
def get_stock_report():
    stock_reports = StockReports(USER_STOCK_WATCH_LIST)
    discounts_update = stock_reports.find_discounted_stocks()
    if discounts_update:
        ChatBotTools.data_store['discounted_stocks'] = discounts_update
        print(f'Discounted stocks: \n{discounts_update}\n')
        SpeechToTextTextToSpeechIO.speak_mainframe(f'Discounted stocks loaded to memory.')
    else:
        SpeechToTextTextToSpeechIO.speak_mainframe(f'No discounted stocks found.')
    recs_update = stock_reports.find_stock_recommendations()
    if recs_update:
        ChatBotTools.data_store['recommended_stocks'] = recs_update
        print(f'Recommended stocks: \n{recs_update}\n')
        SpeechToTextTextToSpeechIO.speak_mainframe(f'Recommended stocks loaded to memory.')
    else:
        SpeechToTextTextToSpeechIO.speak_mainframe(f'No recommended stocks found.')