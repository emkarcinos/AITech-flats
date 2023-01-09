// using HtmlAgilityPack;
// using OpenQA.Selenium.Chrome;
// using OpenQA.Selenium.Chromium;
//
// namespace ImageScrapper.ServiceProviders.WebDriverService;
//
// public class AllegroWebDriverService : IWebDriverService
// {
//     private readonly ChromiumDriver _chromeDriver;
//
//     public AllegroWebDriverService()
//     {
//         var path = "/Users/adamwojdyla/Documents/Private/Priv/ResellScrapperV3/ResellScrapperV3/WebDriver";
//         var options = new ChromeOptions();
//         options.AddArguments(new List<string> { "disable-gpu" });
//
//         _chromeDriver = new ChromeDriver(path, options);
//         _chromeDriver.Manage().Timeouts().ImplicitWait = TimeSpan.FromSeconds(10);
//     }
//
//     public HtmlDocument GetPage(string pageUrl)
//     {
//         _chromeDriver.Navigate().GoToUrl(pageUrl);
//         var doc = new HtmlDocument();
//         doc.LoadHtml(_chromeDriver.PageSource);
//         return doc;
//     }
// }