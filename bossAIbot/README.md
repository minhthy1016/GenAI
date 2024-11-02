Suppose these are sales data and HR data from Company MASHALLAH Groups which business diversified from real estate development-agent sale, multimodal vehicle dealership and funiture supplies for home, offices and corporates from 2001 to 2022. 
The multimodal vehicle dealership under AsamaMove Ltd. is a new business since 2020, while other business are since 2000s.
The data taken from above resources has been adjusted to fit with company business.

You are an application design engineer for AI at a company with two departments:

HR: Human Resource Management.  
Finance: Financial/Accounting Management.  

## 1) Build a separate HRAssistant for the HR department.

Suggestion: First, you need to create a Vector Database consisting of the company's employee information. Then build a RAG pipeline on this database. Test its effectiveness with questions about HR.

## 2) Build a separate FinAssistant for the Finance department.

Suggestion: You also need to prepare a Vector Database for finance first. Then build a RAG pipeline on this database.

## 3) Use routing techniques to combine HRAssistant and FinAssistant into a single BossAssistant so that it can answer all questions about HR and Finance.
