using HtmlAgilityPack;

namespace ImageScrapper.ServiceProviders.WebDriverService;

public interface IWebDriverService
{
    HtmlDocument GetPage(string pageUrl);
}